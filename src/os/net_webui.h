// provenance: original
// Copyright (c) 2026 John Lipka and the Pelias contributors
//
// src/os/net_webui.h: All-in-One Lithic-to-Voxel GUI & HTTP/1.1 Server (Port 8062), Hardware Probe, Continuous Model
// Recognizer, Planes/Tier Quantizer & Trainer Orchestrator. Lives in src/os/ where socket and OS headers belong (BC6).
#pragma once

#include <algorithm>
#include <charconv>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <string_view>
#include <vector>

#if defined(_WIN32)
#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>
#else
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

namespace pelias::os {

struct HttpRequest {
  std::string method;
  std::string target;
  std::string body;
  std::string host;
  std::string origin;
  std::string content_type;
};

struct HttpResponse {
  int status = 200;
  std::string content_type = "application/json; charset=utf-8";
  std::string body;
};

inline std::string lower_ascii(std::string_view text) {
  std::string out(text);
  for (char& c : out) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  return out;
}

inline std::string trim_http_value(std::string_view text) {
  while (!text.empty() && (text.front() == ' ' || text.front() == '\t')) text.remove_prefix(1);
  while (!text.empty() && (text.back() == ' ' || text.back() == '\t')) text.remove_suffix(1);
  return std::string(text);
}

inline bool parse_http_head(std::string_view wire, std::size_t head_end, HttpRequest& request, std::size_t& body_size, int& status) {
  constexpr std::size_t kMaxHeaderBytes = 16 * 1024;
  constexpr std::size_t kMaxBodyBytes = 64 * 1024;
  status = 400;
  if (head_end > kMaxHeaderBytes || head_end > wire.size()) {
    status = 431;
    return false;
  }
  const std::size_t first_end = wire.find("\r\n");
  if (first_end == std::string_view::npos || first_end >= head_end) return false;
  const std::string_view first = wire.substr(0, first_end);
  const std::size_t first_space = first.find(' ');
  const std::size_t second_space = first_space == std::string_view::npos ? first_space : first.find(' ', first_space + 1);
  if (first_space == std::string_view::npos || second_space == std::string_view::npos ||
      (first.substr(second_space + 1) != "HTTP/1.1" && first.substr(second_space + 1) != "HTTP/1.0"))
    return false;
  request.method.assign(first.substr(0, first_space));
  request.target.assign(first.substr(first_space + 1, second_space - first_space - 1));
  if (request.method.empty() || request.target.empty()) return false;

  body_size = 0;
  bool saw_content_length = false;
  std::size_t at = first_end + 2;
  while (at < head_end) {
    const std::size_t end = wire.find("\r\n", at);
    if (end == std::string_view::npos || end > head_end) return false;
    const std::string_view line = wire.substr(at, end - at);
    const std::size_t colon = line.find(':');
    if (colon == std::string_view::npos || colon == 0) return false;
    const std::string name = lower_ascii(line.substr(0, colon));
    const std::string value = trim_http_value(line.substr(colon + 1));
    if (name == "content-length") {
      if (saw_content_length || value.empty()) return false;
      std::size_t n = 0;
      const auto parsed = std::from_chars(value.data(), value.data() + value.size(), n);
      if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size()) return false;
      if (n > kMaxBodyBytes) {
        status = 413;
        return false;
      }
      body_size = n;
      saw_content_length = true;
    } else if (name == "transfer-encoding") {
      status = 501;
      return false;
    } else if (name == "host") {
      request.host = value;
    } else if (name == "origin") {
      request.origin = value;
    } else if (name == "content-type") {
      request.content_type = value;
    }
    at = end + 2;
  }
  return true;
}

inline bool parse_http_request(std::string_view wire, HttpRequest& request, int& status) {
  const std::size_t end = wire.find("\r\n\r\n");
  if (end == std::string_view::npos) {
    status = 400;
    return false;
  }
  std::size_t body_size = 0;
  if (!parse_http_head(wire, end, request, body_size, status)) return false;
  const std::size_t body_at = end + 4;
  if (wire.size() < body_at + body_size) {
    status = 400;
    return false;
  }
  request.body.assign(wire.substr(body_at, body_size));
  return true;
}

inline const char* kPeliasLithicGuiHtml = R"HTML(<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>ΠELIAS v58c // Lithic-to-Voxel Unified AI System</title>
<style>
  :root {
    --slate:#262A31; --well:#1D2026; --light:#C6CCD4; --mid:#899099; --shadow:#4E535B;
    --cyan:#6CD4FF; --core:#C4F7FF; --art:#7FE3EA; --amber:#FFB300; --text:#E6E9ED; --dim:#8E959E;
  }
  * { box-sizing:border-box; margin:0; padding:0; font-family:'Segoe UI',system-ui,sans-serif; }
  body { background:var(--slate); color:var(--text); display:flex; flex-direction:column; height:100vh; overflow:hidden; }
  header { background:var(--well); padding:14px 24px; display:flex; justify-content:space-between; align-items:center; border-bottom:2px solid var(--shadow); }
  .logo { font-size:21px; font-weight:800; letter-spacing:3px; color:var(--core); text-shadow:0 0 12px rgba(108,212,255,0.45); }
  .spear-wrap { background:var(--well); padding:10px 24px; border-bottom:1px solid var(--shadow); }
  .spear-bar { height:22px; background:linear-gradient(90deg,#4E535B,#899099,#3E434C); border:1px solid var(--light); position:relative; }
  .spear-fill { height:100%; width:0; background:linear-gradient(90deg,#1D4E68,var(--art) 85%,var(--core)); box-shadow:0 0 16px var(--cyan); transition:width .25s ease; }
  .spear-meta { display:flex; justify-content:space-between; font-size:12px; color:var(--cyan); margin-top:5px; font-family:monospace; }
  main { flex:1; display:grid; grid-template-columns:360px 1fr; gap:16px; padding:16px; overflow:hidden; }
  .card { background:var(--well); border:1px solid var(--shadow); padding:16px; display:flex; flex-direction:column; gap:10px; overflow-y:auto; }
  h3 { font-size:12px; letter-spacing:1.5px; text-transform:uppercase; color:var(--cyan); border-bottom:1px solid #2F353F; padding-bottom:5px; }
  label { font-size:12px; color:var(--dim); display:flex; justify-content:space-between; }
  select, input[type="text"], input[type="number"] { width:100%; background:#14171B; color:var(--text); border:1px solid var(--shadow); padding:8px; font-size:13px; }
  .grid { display:grid; grid-template-columns:1fr 1fr; gap:8px; }
  .stat { background:#15181E; padding:10px; border-left:3px solid var(--cyan); }
  .val { font-size:18px; font-weight:700; color:var(--core); font-family:monospace; }
  .lbl { font-size:10px; color:var(--dim); text-transform:uppercase; }
  button { background:#1C3B4F; color:var(--core); border:1px solid var(--cyan); padding:9px 14px; cursor:pointer; font-weight:700; text-transform:uppercase; letter-spacing:1px; }
  button:hover { background:#27536E; }
  #console { flex:1; background:#14171B; padding:14px; font-family:monospace; white-space:pre-wrap; overflow-y:auto; line-height:1.45; }
  .row { display:flex; gap:8px; flex-wrap:wrap; }
  button:disabled { opacity:.45; cursor:not-allowed; }
</style>
</head>
<body>
<header>
  <div class="logo">ΠELIAS // LITHIC-TO-VOXEL ALL-IN-ONE SYSTEM v58c</div>
  <div class="row">
    <button id="probe-button" onclick="runCmd('probe')">Hardware & Memory Probe</button>
    <button id="engine-button" onclick="runCmd('engine-check')">Run Engine Gate</button>
    <button id="quantize-button" onclick="runCmd('quantize')">Quantize Model</button>
  </div>
</header>
<div class="spear-wrap">
  <div class="spear-bar"><div class="spear-fill" id="spear"></div></div>
  <div class="spear-meta">
    <span id="status">Connecting to Pelias...</span>
    <span id="port-label">LOCAL SERVICE</span>
  </div>
</div>
<main>
  <div class="card">
    <h3>Model & Decode Settings</h3>
    <input type="text" id="model-path" placeholder="Path to a Qwen35MoE GGUF model">
    <input type="text" id="output-path" placeholder="Quantized output GGUF path (required for quantize)">
    <select id="quant-type">
      <option>Q4_K_M</option><option>Q6_K</option><option>Q8_0</option><option>BF16</option><option>F16</option><option>F32</option>
    </select>
    <h3>GPU Engine & MTP</h3>
    <label>Sliding Bit-Planes (0 = off; 2–5 bits) <span id="b-val">Off</span></label>
    <input type="range" id="plane-bits" min="0" max="4" value="0" oninput="updateControls()">
    <label>MTP Draft Tokens (--mtp-draft) <span id="m-val">0</span></label>
    <input type="range" id="mtp-draft" min="0" max="4" value="0" oninput="updateControls()">
    <label>Maximum generated tokens <span></span></label>
    <input type="number" id="max-tokens" min="1" max="1024" value="128">
    <div class="grid">
      <div class="stat"><div class="val" id="st-backend">—</div><div class="lbl">Build Backend</div></div>
      <div class="stat"><div class="val" id="st-plane">Off</div><div class="lbl">Plane Mode</div></div>
      <div class="stat"><div class="val" id="st-mtp">0</div><div class="lbl">MTP Draft Tokens</div></div>
      <div class="stat"><div class="val" id="st-model">Not loaded</div><div class="lbl">Model State</div></div>
    </div>
  </div>
  <div class="card">
    <h3>Pelias Operations</h3>
    <div id="console">Connecting to the local Pelias process. Chat and quantize run through the existing CLI/model implementation. Training is not implemented in this build.</div>
    <div class="row">
      <input type="text" id="prompt" placeholder="Enter a prompt..." onkeydown="if(event.key==='Enter')sendChat()">
      <button onclick="sendChat()">Send</button>
    </div>
  </div>
</main>
<script>
let busy = false;
let planeBitsAvailable = false;
function selectedPlaneBits() {
  const value = Number(document.getElementById('plane-bits').value);
  return value === 0 ? 0 : value + 1;
}
function updateControls() {
  const bits = selectedPlaneBits();
  document.getElementById('b-val').textContent = bits === 0 ? 'Off' : bits + '-bit';
  document.getElementById('st-plane').textContent = bits === 0 ? 'Off' : bits + '-bit';
  document.getElementById('m-val').textContent = document.getElementById('mtp-draft').value;
  document.getElementById('st-mtp').textContent = document.getElementById('mtp-draft').value;
}
function setBusy(value) {
  busy = value;
  for (const id of ['probe-button', 'engine-button', 'quantize-button']) document.getElementById(id).disabled = value || (id === 'engine-button' && !planeBitsAvailable);
  for (const id of ['model-path', 'output-path', 'quant-type', 'plane-bits', 'mtp-draft', 'max-tokens', 'prompt'])
    document.getElementById(id).disabled = value || (id === 'plane-bits' && !planeBitsAvailable);
  document.querySelector('#console-row button').disabled = value;
  document.getElementById('spear').style.width = value ? '35%' : '100%';
}
function appendOutput(text) {
  const con = document.getElementById('console');
  con.textContent += '\n' + text + '\n';
  con.scrollTop = con.scrollHeight;
}
async function loadStatus() {
  try {
    const r = await fetch('/api/status');
    const s = await r.json();
    planeBitsAvailable = !!s.plane_bits_available;
    document.getElementById('st-backend').textContent = s.backend;
    document.getElementById('status').textContent = 'Pelias ' + s.version + ' — ' + s.backend + (s.gpu_available ? ' / GPU ready' : ' / GPU engine unavailable');
    document.getElementById('port-label').textContent = 'PORT ' + (location.port || '80') + ' // LOCAL';
    document.getElementById('spear').style.width = '100%';
    document.getElementById('plane-bits').disabled = !planeBitsAvailable;
    document.getElementById('engine-button').disabled = !planeBitsAvailable;
  } catch (e) {
    document.getElementById('status').textContent = 'Pelias API unavailable';
    appendOutput('Could not connect to Pelias: ' + e.message);
  }
}
async function runCmd(command) {
  if (busy) return;
  const payload = {command, model: document.getElementById('model-path').value.trim(), plane_bits: selectedPlaneBits()};
  if (command === 'quantize') {
    payload.output = document.getElementById('output-path').value.trim();
    payload.type = document.getElementById('quant-type').value;
  }
  setBusy(true);
  document.getElementById('status').textContent = 'Running ' + command + '...';
  try {
    const r = await fetch('/api/cmd', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload)});
    const data = await r.json();
    if (data.error) appendOutput('[ERROR] ' + data.error.message);
    else {
      appendOutput('[' + data.command + ', exit ' + data.exit_code + ']\n' + data.output);
      if (command === 'engine-check' && data.ok) document.getElementById('st-model').textContent = 'Engine ready';
    }
  } catch (e) {
    appendOutput('[ERROR] ' + e.message);
  } finally {
    setBusy(false);
    document.getElementById('status').textContent = 'Pelias ready';
  }
}
async function sendChat() {
  if (busy) return;
  const inp = document.getElementById('prompt');
  const prompt = inp.value.trim();
  if (!prompt) return;
  const payload = {
    model: document.getElementById('model-path').value.trim(), prompt,
    plane_bits: selectedPlaneBits(), mtp_draft: Number(document.getElementById('mtp-draft').value),
    max_tokens: Number(document.getElementById('max-tokens').value)
  };
  inp.value = '';
  appendOutput('> ' + prompt);
  setBusy(true);
  document.getElementById('status').textContent = 'Generating...';
  try {
    const r = await fetch('/v1/chat/completions', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload)});
    const data = await r.json();
    if (!r.ok || data.error) appendOutput('[ERROR] ' + (data.error?.message || ('HTTP ' + r.status)));
    else {
      appendOutput(data.choices?.[0]?.message?.content || '[No text returned]');
      document.getElementById('st-model').textContent = 'Loaded for request';
    }
  } catch (e) {
    appendOutput('[ERROR] ' + e.message);
  } finally {
    setBusy(false);
    document.getElementById('status').textContent = 'Pelias ready';
  }
}
document.getElementById('prompt').parentElement.id = 'console-row';
document.getElementById('mtp-draft').addEventListener('input', updateControls);
document.getElementById('plane-bits').addEventListener('input', updateControls);
updateControls();
loadStatus();
</script>
</body>
</html>)HTML";

inline void launch_browser_app(int port) {
  const std::string url = "http://127.0.0.1:" + std::to_string(port) + "/";
#if defined(_WIN32)
  const std::string cmd = "start \"\" msedge --app=\"" + url + "\" || start \"\" \"" + url + "\"";
  std::system(cmd.c_str());
#elif defined(__linux__)
  const std::string cmd = "xdg-open " + url + " >/dev/null 2>&1 &";
  if (std::system(cmd.c_str()) != 0) {}
#endif
}

inline const char* http_reason(int status) {
  switch (status) {
    case 200: return "OK";
    case 400: return "Bad Request";
    case 403: return "Forbidden";
    case 404: return "Not Found";
    case 405: return "Method Not Allowed";
    case 413: return "Payload Too Large";
    case 415: return "Unsupported Media Type";
    case 431: return "Request Header Fields Too Large";
    case 500: return "Internal Server Error";
    case 501: return "Not Implemented";
    default: return "Error";
  }
}

inline HttpResponse http_error(int status, const std::string& message) {
  return {status, "application/json; charset=utf-8", "{\"error\":\"" + message + "\"}"};
}

inline bool local_gui_origin(const HttpRequest& request, int port) {
  const std::string address = "127.0.0.1:" + std::to_string(port);
  const std::string localhost = "localhost:" + std::to_string(port);
  if (!request.host.empty() && request.host != address && request.host != localhost) return false;
  if (!request.origin.empty() && request.origin != "http://" + address && request.origin != "http://" + localhost) return false;
  return true;
}

inline bool send_http_response(
#if defined(_WIN32)
    SOCKET client,
#else
    int client,
#endif
    const HttpResponse& response) {
  std::string header = "HTTP/1.1 " + std::to_string(response.status) + " " + http_reason(response.status) +
                       "\r\nContent-Type: " + response.content_type + "\r\nContent-Length: " + std::to_string(response.body.size()) +
                       "\r\nConnection: close\r\nCache-Control: no-store\r\nX-Content-Type-Options: nosniff\r\n\r\n";
  auto send_all = [&](const char* data, std::size_t size) {
    std::size_t sent = 0;
    while (sent < size) {
      const int chunk = static_cast<int>(std::min<std::size_t>(size - sent, 1u << 20));
#if defined(_WIN32)
      const int n = send(client, data + sent, chunk, 0);
#else
      const int n = static_cast<int>(send(client, data + sent, static_cast<std::size_t>(chunk), 0));
#endif
      if (n <= 0) return false;
      sent += static_cast<std::size_t>(n);
    }
    return true;
  };
  return send_all(header.data(), header.size()) && send_all(response.body.data(), response.body.size());
}

template <typename ApiHandler>
inline int serve_http_loop(int port, bool open_gui, const std::string& health_json, ApiHandler api_handler) {
#if defined(_WIN32)
  WSADATA wsa;
  if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0) return 1;
  SOCKET srv = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
  if (srv == INVALID_SOCKET) {
    WSACleanup();
    return 1;
  }
#else
  int srv = socket(AF_INET, SOCK_STREAM, 0);
  if (srv < 0) return 1;
  int opt = 1;
  setsockopt(srv, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
#endif
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(static_cast<unsigned short>(port));
  addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  if (bind(srv, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0 || listen(srv, 16) != 0) {
    std::fprintf(stderr, "pelias serve: port %d is unavailable\n", port);
#if defined(_WIN32)
    closesocket(srv);
    WSACleanup();
#else
    close(srv);
#endif
    return 1;
  }
  std::fprintf(stderr, "pelias serve: listening on http://127.0.0.1:%d/ (Lithic-to-Voxel All-in-One UI + OpenAI API)\n", port);
  if (open_gui) launch_browser_app(port);
  for (;;) {
    sockaddr_in cli_addr{};
#if defined(_WIN32)
    int cli_len = sizeof(cli_addr);
    SOCKET cli = accept(srv, reinterpret_cast<sockaddr*>(&cli_addr), &cli_len);
    if (cli == INVALID_SOCKET) continue;
#else
    socklen_t cli_len = sizeof(cli_addr);
    int cli = accept(srv, reinterpret_cast<sockaddr*>(&cli_addr), &cli_len);
    if (cli < 0) continue;
#endif
    constexpr std::size_t kMaxHeaderBytes = 16 * 1024;
    constexpr std::size_t kMaxBodyBytes = 64 * 1024;
    std::string wire;
    HttpRequest request;
    HttpResponse response;
    bool parsed = false;
    int parse_status = 400;
    while (wire.size() <= kMaxHeaderBytes + kMaxBodyBytes + 4) {
      char chunk[4096];
#if defined(_WIN32)
      const int n = recv(cli, chunk, sizeof(chunk), 0);
#else
      const int n = static_cast<int>(recv(cli, chunk, sizeof(chunk), 0));
#endif
      if (n <= 0) break;
      wire.append(chunk, static_cast<std::size_t>(n));
      const std::size_t head_end = wire.find("\r\n\r\n");
      if (head_end == std::string::npos) {
        if (wire.size() > kMaxHeaderBytes) parse_status = 431;
        if (parse_status == 431) break;
        continue;
      }
      std::size_t body_size = 0;
      HttpRequest head;
      if (!parse_http_head(wire, head_end, head, body_size, parse_status)) break;
      if (wire.size() >= head_end + 4 + body_size) {
        parsed = parse_http_request(wire, request, parse_status);
        break;
      }
    }
    if (!parsed) {
      if (wire.size() > kMaxHeaderBytes + kMaxBodyBytes + 4 && parse_status == 400) parse_status = 413;
      response = http_error(parse_status, "Invalid or oversized HTTP request");
    } else if (!local_gui_origin(request, port)) {
      response = http_error(403, "GUI API accepts requests only from this local Pelias page");
    } else if (request.method == "GET" && request.target == "/") {
      response = {200, "text/html; charset=utf-8", kPeliasLithicGuiHtml};
    } else if (request.method == "GET" && request.target == "/health") {
      response = {200, "application/json; charset=utf-8", health_json};
    } else if ((request.method == "GET" && request.target == "/api/status") ||
               (request.method == "POST" && (request.target == "/api/cmd" || request.target == "/v1/chat/completions"))) {
      try {
        response = api_handler(request);
      } catch (...) {
        response = http_error(500, "GUI operation failed unexpectedly");
      }
    } else if (request.target.rfind("/api/", 0) == 0 || request.target.rfind("/v1/", 0) == 0) {
      response = http_error(405, "Unsupported GUI API method or route");
    } else {
      response = http_error(404, "Page not found");
    }
    send_http_response(cli, response);
#if defined(_WIN32)
    closesocket(cli);
#else
    close(cli);
#endif
  }
}

}  // namespace pelias::os
