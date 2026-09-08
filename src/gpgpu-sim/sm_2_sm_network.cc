#include "sm_2_sm_network.h"
#include <algorithm>
#include <unordered_set>
#include <utility>
#include "gpu-sim.h"

int sm2sm_latency = 171;  // Default latency for SM to SM network
int sm2sm_bandwidth =
    88;  // Default bandwidth for SM to SM network in bits/cycle

// N->1 output-port bandwidth-collapse model (calibrated to Dissecting-Hopper aggregate SM-to-SM
// bw: 3948/2332/1960 GB/s for cluster 2/4/8). collapse_factor(k) = effective-bw fraction when k
// concurrent inputs contend for one output port. k<=1 => 1.0 (baseline unchanged).
int sm2sm_collapse = 0;  // 0 = stock crossbar; 1 = enable calibrated collapse
// DSM fan-in calibration (2026-09-04). Real H20 DSM read latency is FLAT in cluster
// size (validation_crossover.dat: 7299/7445/7427 cyc at nfloats=1024, c2/c4/c8) but the
// stock single-winner-per-output-port RR below serializes all k consumers of one producer,
// making sim DSM grow 8072->10504. The producer SMEM on real Hopper sustains 71-103 B/cyc
// under broadcast (ClusterSim IISWC'25 Sec.V-B), i.e. it services several readers per cycle,
// while one output port here delivers only m_bandwidth=88 bit/cyc to a single winner.
// S = number of distinct requesters an output port may grant per cycle. S<=1 is EXACTLY the
// stock path (bit-identical); S>1 relaxes the fan-in serialization only.
int sm2sm_service_slots = 1;  // 1 = stock (unchanged); >1 = calibrated fan-in service
static inline double sm2sm_collapse_factor(int k) {
  if (k <= 1) return 1.0;
  // pinned from the bw-cliff microbench (orthogonal to the payload crossover):
  //   k=2 -> 1.00, k=4 -> 2332/3948 = 0.591, k=8 -> 1960/3948 = 0.496.
  // piecewise-linear in k between the measured anchors, clamped.
  if (k <= 2) return 1.00;
  if (k <= 4) return 1.00 + (0.591 - 1.00) * (k - 2) / (4 - 2);
  if (k <= 8) return 0.591 + (0.496 - 0.591) * (k - 4) / (8 - 4);
  return 0.496;  // >=8 contenders: floor
}

void sm2sm_network_options(class OptionParser* opp) {
  option_parser_register(opp, "-sm_2_sm_network_latency", OPT_INT32,
                         &sm2sm_latency, "Latency of the SM to SM network",
                         "171");
  option_parser_register(
      opp, "-sm_2_sm_network_bw", OPT_INT32, &sm2sm_bandwidth,
      "Bandwidth of the SM to SM network in bit/cycle", "88");
  option_parser_register(
      opp, "-sm_2_sm_collapse", OPT_INT32, &sm2sm_collapse,
      "Enable N->1 output-port bandwidth-collapse model (0=stock,1=calibrated)", "0");
  option_parser_register(
      opp, "-sm_2_sm_service_slots", OPT_INT32, &sm2sm_service_slots,
      "Requesters an output port may grant per cycle (1=stock RR, >1=calibrated fan-in)",
      "1");
}

cluster_shmem_request::cluster_shmem_request(warp_inst_t* warp, addr_t address,
                                             bool is_write, bool is_atomic,
                                             unsigned origin_shader_id,
                                             unsigned target_shader_id,
                                             unsigned tid, unsigned latency,
                                             unsigned size)
    : m_warp(warp), m_address(address) {
  m_is_write = is_write;
  m_is_atomic = is_atomic;
  m_target_shader_id = target_shader_id;
  m_origin_shader_id = origin_shader_id;
  m_tid = tid;
  m_is_send = false;
  m_is_response = false;
  m_is_complete = false;
  m_latency = latency;
  m_size = size;
}

SM_2_SM_network::SM_2_SM_network(unsigned n_shader,
                                 const class shader_core_config* config,
                                 const class gpgpu_sim* gpu) {
  m_n_shader = n_shader;
  m_config = config;
  m_gpu = gpu;
}

Crossbar::Crossbar(unsigned n_shader, const class shader_core_config* config,
                   const class gpgpu_sim* gpu)
    : SM_2_SM_network(n_shader, config, gpu),
      rr_pointers(n_shader, 0),
      input_queues(n_shader),
      output_queues(n_shader),
      m_bandwidth(sm2sm_bandwidth),
      m_latency(sm2sm_latency),
      m_time(0) {}

void Crossbar::Push(unsigned input_deviceID, unsigned output_deviceID,
                    std::shared_ptr<cluster_shmem_request> data,
                    unsigned int size, Interconnect_type network) {
  output_deviceID = sid_to_gid(output_deviceID);
  input_deviceID = sid_to_gid(input_deviceID);
  if (data.get()->is_atomic) {
    size = data->size * 8 * 4;
  } else {
    size = data->size * 8;
  }

  assert(data.get() != nullptr);
  input_queues[input_deviceID].emplace(input_deviceID, output_deviceID, size,
                                       m_time, data);

  // output_queues[output_deviceID].emplace(data);
}

std::shared_ptr<cluster_shmem_request> Crossbar::Pop(
    unsigned ouput_deviceID, Interconnect_type network) {
  ouput_deviceID = sid_to_gid(ouput_deviceID);
  if (!output_queues[ouput_deviceID].empty()) {
    auto result = output_queues[ouput_deviceID].front();
    output_queues[ouput_deviceID].pop();
    return result;
  }
  return nullptr;
}

void Crossbar::Advance() {
  // Quit early if no messages
  bool empty = true;
  for (const auto& node : input_queues) {
    if (!node.empty()) {
      empty = false;
      break;
    }
  }
  if (in_flight.empty() && empty) return;

  const int num_ports = m_n_shader;
  // std::cout << "Time step: " << m_time << "\n";

  // Check in-flight messages
  for (size_t i = 0; i < in_flight.size(); ++i) {
    if (m_time - in_flight[i].second >= m_latency) {
      // std::cout << "Delivered message from " << in_flight[i].first.src << "
      // to "
      //           << in_flight[i].first.dst << "\n";

      auto& data = in_flight[i].first.data;
      const auto dst = in_flight[i].first.dst;
      assert(sid_to_gid(data->target_shader_id) == dst);
      assert(!data->complete);
      data->complete = true;
      output_queues[dst].push(data);
      in_flight.erase(in_flight.begin() + i);
      i = 0;
    }
  }

  // Build request lists for each output
  std::vector<std::vector<int>> requests(num_ports);
  for (int i = 0; i < num_ports; ++i) {
    if (!input_queues[i].empty()) {
      int dst = input_queues[i].front().dst;
      requests[dst].push_back(i);
    }
  }

  // 3. Arbitration per output port.
  // collapse==2: SHARED-BW-POOL model — the output port's total bandwidth m_bandwidth is SPLIT
  // among all k concurrent contenders (each gets m_bandwidth/k), draining ALL of them this cycle.
  // This makes a large message under k-way contention take ~k x longer (aggregate bw does not grow
  // with contenders), reproducing the silicon N->1 bandwidth collapse in a payload-DEPENDENT way.
  if (sm2sm_collapse == 2) {
    for (int out = 0; out < num_ports; ++out) {
      int k = (int)requests[out].size();
      if (k == 0) continue;
      { static long _kh[16]={0}; if(k<16)_kh[k]++; static long _n=0;
        if((++_n)%200000==1) fprintf(stderr,"[KHIST] k1=%ld k2=%ld k3=%ld k4=%ld k5=%ld k6=%ld k7=%ld k8=%ld\n",_kh[1],_kh[2],_kh[3],_kh[4],_kh[5],_kh[6],_kh[7],_kh[8]); }
      int per = std::max(1, m_bandwidth / k);   // shared pool split among k contenders
      for (int requester : requests[out]) {
        Message& msg = input_queues[requester].front();
        int send_amount = std::min(per, msg.size - msg.sent_bits);
        msg.sent_bits += send_amount;
        if (msg.sent_bits >= msg.size) {
          in_flight.push_back({msg, m_time});
          input_queues[requester].pop();
        }
      }
    }
    ++m_time;
    return;
  }

  // 3b. Stock round-robin arbitration per output port (collapse 0=off, 1=scalar-factor)
  // The slot loop below runs EXACTLY ONCE when sm2sm_service_slots <= 1, in which case this
  // is the unmodified stock code path (one grant per output port per cycle). See the note at
  // the sm2sm_service_slots definition for the silicon justification of S > 1.
  const int n_slots = (sm2sm_service_slots > 1) ? sm2sm_service_slots : 1;
  for (int out = 0; out < num_ports; ++out) {
    std::vector<int> granted;  // empty and unused when n_slots == 1
    for (int slot = 0; slot < n_slots; ++slot) {
    if (!requests[out].empty()) {
      int chosen = -1;
      int start = rr_pointers[out];

      for (int i = 0; i < num_ports; ++i) {
        int idx = (start + i) % num_ports;
        for (int requester : requests[out]) {
          if (requester == idx) {
            // a requester may be granted at most one slot per cycle
            if (n_slots > 1 &&
                std::find(granted.begin(), granted.end(), idx) != granted.end())
              continue;
            chosen = idx;
            rr_pointers[out] = (idx + 1) % num_ports;
            break;
          }
        }
        if (chosen != -1) break;
      }

      if (chosen != -1) {
        if (n_slots > 1) granted.push_back(chosen);
        Message& msg = input_queues[chosen].front();
        int k = (int)requests[out].size();
        int eff_bw = (sm2sm_collapse == 1)
            ? std::max(1, (int)(m_bandwidth * sm2sm_collapse_factor(k)))
            : m_bandwidth;
        int send_amount = std::min(eff_bw, msg.size - msg.sent_bits);
        msg.sent_bits += send_amount;

        if (msg.sent_bits >= msg.size) {
          in_flight.push_back({msg, m_time});
          input_queues[chosen].pop();
        }
      } else {
        break;  // nothing left to grant on this port this cycle
      }
    }
    }
  }
  ++m_time;
}

IdealNetwork::IdealNetwork(unsigned n_shader, const shader_core_config* config,
                           const gpgpu_sim* gpu)
    : SM_2_SM_network(n_shader, config, gpu),
      input_queues(n_shader),
      output_queues(n_shader),
      m_time(0),
      m_latency(sm2sm_latency) {}

void IdealNetwork::Push(unsigned input_deviceID, unsigned output_deviceID,
                        std::shared_ptr<cluster_shmem_request> data,
                        unsigned int size, Interconnect_type network) {
  output_deviceID = sid_to_gid(output_deviceID);
  input_deviceID = sid_to_gid(input_deviceID);
  size = data->size * 8;
  assert(data.get() != nullptr);
  input_queues[input_deviceID].emplace(input_deviceID, output_deviceID, size,
                                       m_time, data);
}

void IdealNetwork::Advance() {
  // Quit early if no messages
  bool empty = true;
  for (const auto& node : input_queues) {
    if (!node.empty()) {
      empty = false;
      break;
    }
  }
  if (empty) return;

  // Process all input queues

  for (int i = 0; i < m_n_shader; i++) {
    if (!input_queues[i].empty()) {
      auto& msg = input_queues[i].front();
      if (m_time - msg.time_injected > m_latency) {
        output_queues[msg.dst].push(msg.data);
        input_queues[i].pop();
      }
    }
  }
  ++m_time;
}

std::shared_ptr<cluster_shmem_request> IdealNetwork::Pop(
    unsigned ouput_deviceID, Interconnect_type network) {
  ouput_deviceID = sid_to_gid(ouput_deviceID);
  if (!output_queues[ouput_deviceID].empty()) {
    auto result = output_queues[ouput_deviceID].front();
    output_queues[ouput_deviceID].pop();
    return result;
  }
  return nullptr;
}

Ringbus::Ringbus(unsigned n_shader, const class shader_core_config* config,
                 const class gpgpu_sim* gpu)
    : SM_2_SM_network(n_shader, config, gpu),
      output_queues(n_shader),
      num_nodes(n_shader),
      ring_(n_shader),
      m_time(0),
      bandwidth(sm2sm_bandwidth),
      m_latency(sm2sm_latency) {}

void Ringbus::Push(unsigned input_deviceID, unsigned output_deviceID,
                   std::shared_ptr<cluster_shmem_request> data,
                   unsigned int size, Interconnect_type network) {
  output_deviceID = sid_to_gid(output_deviceID);
  input_deviceID = sid_to_gid(input_deviceID);
  Message msg = {input_deviceID, output_deviceID, data,
                 size,           m_latency,       input_deviceID};
  ring_[input_deviceID].push(msg);
}

std::shared_ptr<cluster_shmem_request> Ringbus::Pop(unsigned output_deviceID,
                                                    Interconnect_type network) {
  output_deviceID = sid_to_gid(output_deviceID);
  auto& buf = output_queues[output_deviceID];
  if (buf.empty()) return nullptr;

  auto msg = buf.front();
  buf.pop();
  return msg.data;
}

void Ringbus::Advance() {
  if (!Busy()) return;

  for (int i = 0; i < num_nodes; i++) {
    if (ring_[i].empty()) continue;

    auto& msg = ring_[i].front();

    if (msg.output_id == i) {
      std::cout << "Moved message from " << msg.input_id << " to " << i << "\n";
      output_queues[i].push(ring_[i].front());
      ring_[i].pop();
    } else if (msg.remaining_latency <= 0) {
      const int nn = this->next_node(i);
      msg.current_position = nn;
      msg.remaining_latency = m_latency;
      ring_[nn].push(msg);
      ring_[i].pop();
    } else {
      msg.remaining_latency--;
    }
  }
}

bool Ringbus::Busy() const {
  for (const auto& node : ring_) {
    if (!node.empty()) return true;
  }

  return false;
}