#include "os/net_webui.h"
#include "pelias_test.h"

using pelias::os::HttpRequest;
using pelias::os::parse_http_request;
using pelias::os::local_gui_origin;

TEST(webui_parses_json_post) {
  const std::string body = R"({"prompt":"a\\nb"})";
  const std::string wire = "POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1:8062\r\nOrigin: http://127.0.0.1:8062\r\nContent-Type: application/json\r\nContent-Length: " +
						   std::to_string(body.size()) + "\r\n\r\n" + body;
  HttpRequest request;
  int status = 0;
  CHECK(parse_http_request(wire, request, status));
  CHECK_EQ(request.method, std::string("POST"));
  CHECK_EQ(request.target, std::string("/v1/chat/completions"));
  CHECK_EQ(request.body, body);
  CHECK_EQ(request.host, std::string("127.0.0.1:8062"));
  CHECK_EQ(request.origin, std::string("http://127.0.0.1:8062"));
  CHECK_EQ(request.content_type, std::string("application/json"));
  CHECK(local_gui_origin(request, 8062));
}

TEST(webui_rejects_malformed_and_oversized_requests) {
  HttpRequest request;
  int status = 0;
  CHECK(!parse_http_request("POST /api/cmd HTTP/1.1\r\nContent-Length: nope\r\n\r\n", request, status));
  CHECK_EQ(status, 400);
  CHECK(!parse_http_request("POST /api/cmd HTTP/1.1\r\nContent-Length: 2\r\n\r\nx", request, status));
  CHECK_EQ(status, 400);
  CHECK(!parse_http_request("POST /api/cmd HTTP/1.1\r\nContent-Length: 65537\r\n\r\n", request, status));
  CHECK_EQ(status, 413);
}

TEST(webui_rejects_remote_origins) {
  HttpRequest request;
  request.host = "127.0.0.1:8062";
  request.origin = "https://example.invalid";
  CHECK(!local_gui_origin(request, 8062));
  request.origin = "http://127.0.0.1:8062";
  CHECK(local_gui_origin(request, 8062));
}
