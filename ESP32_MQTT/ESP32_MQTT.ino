// #include <ESP8266WiFi.h>
// #include <PubSubClient.h>

// // Wi-Fi và MQTT server
// const char* ssid = "FreeWifi";
// const char* password = "12345678";
// const char* mqtt_server = "192.168.66.80";  // IP máy tính chạy Mosquitto

// #define Red_led D3
// #define Green_led D4

// WiFiClient espClient;
// PubSubClient client(espClient);

// // ===== Kết nối WiFi =====
// void setup_wifi() {
//   Serial.print("Connecting to ");
//   Serial.println(ssid);
//   WiFi.begin(ssid, password);

//   while (WiFi.status() != WL_CONNECTED) {
//     delay(500);
//     Serial.print(".");
//   }

//   Serial.println("\n✅ WiFi connected");
//   Serial.print("IP address: ");
//   Serial.println(WiFi.localIP());
// }

// // ===== Hàm xử lý khi nhận tin nhắn MQTT =====
// void callback(char* topic, byte* payload, unsigned int length) {
//   String message;
//   for (int i = 0; i < length; i++) {
//     message += (char)payload[i];
//   }

//   Serial.print("📩 Nhận tin nhắn từ topic [");
//   Serial.print(topic);
//   Serial.print("]: ");
//   Serial.println(message);

//   if (message.indexOf(':') != -1) {
//     Serial.println("🔔 Phát hiện dấu ':' → bật D2");
//     delay(1000);  // giữ 1 giây
//     digitalWrite(Red_led, LOW);
//     digitalWrite(Green_led, HIGH);
//   }
// }

// // ===== Kết nối MQTT và gửi message =====
// void reconnect() {
//   while (!client.connected()) {
//     Serial.print("Attempting MQTT connection...");
//     if (client.connect("ESP8266Client")) {
//       Serial.println("connected");

//       // Gửi lệnh "score"
//       client.publish("esp8266/score", "score");
//       Serial.println("✅ Gửi lệnh: score");

//       // Đăng ký nhận phản hồi
//       client.subscribe("Score/finish");
//       Serial.println("📡 Đăng ký nhận phản hồi tại topic: esp8266/response");

//       digitalWrite(Red_led, HIGH);
//       digitalWrite(Green_led, LOW);
//     } else {
//       Serial.print("❌ Kết nối thất bại, mã lỗi = ");
//       Serial.print(client.state());
//       Serial.println(" → thử lại sau 1 giây");
//       delay(1000);
//     }
//   }
// }

// // ===== Setup ban đầu =====
// void setup() {
//   Serial.begin(115200);
//   pinMode(Red_led, OUTPUT);
//   pinMode(Green_led, OUTPUT);

//   digitalWrite(Red_led, LOW);
//   digitalWrite(Green_led, HIGH);

//   setup_wifi();
//   client.setServer(mqtt_server, 1883);
//   client.setCallback(callback);  // ⚠️ Đăng ký hàm nhận message
// }

// // ===== Vòng lặp chính =====
// void loop() {
//   if (!client.connected()) {
//     reconnect();
//   }
//   client.loop();
// }





#include <ESP8266WiFi.h>
#include <PubSubClient.h>

// Wi-Fi và MQTT server
const char* ssid = "FreeWifi";
const char* password = "12345678";
const char* mqtt_server = "192.168.66.80";  // IP máy tính chạy Mosquitto

#define Red_led D3
#define Green_led D4
#define Button_pin D5

WiFiClient espClient;
PubSubClient client(espClient);

// Cờ báo hiệu nút được nhấn (volatile để dùng trong ISR và loop)
volatile bool buttonPressedFlag = false;
// Biến để debounce trong ISR
volatile unsigned long lastInterruptTime = 0;
const unsigned long debounceDelay = 300; // 50ms debounce

// ===== Kết nối WiFi =====
void setup_wifi() {
  Serial.print("Connecting to ");
  Serial.println(ssid);
  WiFi.begin(ssid, password);

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  Serial.println("\n✅ WiFi connected");
  Serial.print("IP address: ");
  Serial.println(WiFi.localIP());
}

// ===== Hàm xử lý khi nhận tin nhắn MQTT =====
void callback(char* topic, byte* payload, unsigned int length) {
  String message;
  for (unsigned int i = 0; i < length; i++) {
    message += (char)payload[i];
  }

  Serial.print("📩 Nhận tin nhắn từ topic [");
  Serial.print(topic);
  Serial.print("]: ");
  Serial.println(message);

  if (message.indexOf(':') != -1) {
    Serial.println("🔔 Phát hiện dấu ':' → bật Green_led");
    digitalWrite(Red_led, LOW);
    digitalWrite(Green_led, HIGH);
    delay(1000);  // Giữ trong 1 giây
  }
}

// ===== Kết nối MQTT và subscribe =====
void reconnect() {
  while (!client.connected()) {
    Serial.print("Attempting MQTT connection...");
    if (client.connect("ESP8266Client")) {
      Serial.println("connected");
      // Subscribe vào topic để nhận phản hồi
      client.subscribe("Score/finish");
      Serial.println("📡 Subscribed to: Score/finish");
    } else {
      Serial.print("❌ Kết nối thất bại, mã lỗi = ");
      Serial.print(client.state());
      Serial.println(" → Thử lại sau 1 giây");
      delay(1000);
    }
  }
}

// ===== ISR xử lý ngắt nút nhấn =====
void IRAM_ATTR handleButtonInterrupt() {
  unsigned long interruptTime = millis();
  // Debounce: chỉ xử lý nếu khoảng cách giữa 2 ngắt lớn hơn debounceDelay
  if (interruptTime - lastInterruptTime > debounceDelay) {
    buttonPressedFlag = true;
  }
  lastInterruptTime = interruptTime;
}

// ===== Setup ban đầu =====
void setup() {
  Serial.begin(115200);

  pinMode(Red_led, OUTPUT);
  pinMode(Green_led, OUTPUT);
  pinMode(Button_pin, INPUT_PULLUP);  // Sử dụng pull-up nội bộ cho nút

  digitalWrite(Red_led, LOW);
  digitalWrite(Green_led, HIGH);

  setup_wifi();
  client.setServer(mqtt_server, 1883);
  client.setCallback(callback);

  // Thiết lập ngắt cho nút nhấn: kích hoạt khi tín hiệu rơi (FALLING)
  attachInterrupt(digitalPinToInterrupt(Button_pin), handleButtonInterrupt, FALLING);
}

// ===== Vòng lặp chính =====
void loop() {
  if (!client.connected()) {
    reconnect();
  }
  client.loop();

  // Kiểm tra cờ nút được nhấn bởi ISR
  if (buttonPressedFlag) {
    // Reset cờ trước khi xử lý
    buttonPressedFlag = false;
    Serial.println("🖲️ Button nhấn → gửi lệnh score");
    client.publish("esp8266/score", "score");
    digitalWrite(Red_led, HIGH);
    digitalWrite(Green_led, LOW);
  }
}
