#include <PubSubClient.h>

#include <ESP8266WiFi.h>

const char* ssid = "begy";
const char* password = "00000010";
const char* mqtt_server = "192.168.186.210";

WiFiClient espClient;
PubSubClient client(espClient);

const int ledPin = 2; // Onboard LED for most ESP8266
bool dc_trigger;
volatile long encoder = 0;

void setup_wifi() {
  Serial.print("Connecting to ");
  Serial.println(ssid);
  WiFi.begin(ssid, password);

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  Serial.println("\nWiFi connected");
  Serial.print("IP address: ");
  Serial.println(WiFi.localIP());
}


void callback(char* topic, byte* payload, unsigned int length) {
  Serial.print("Message arrived [");
  Serial.print(topic);
  Serial.print("]: ");
  
  String message;
  for (int i = 0; i < length; i++) {
    message += (char)payload[i];
  }

  if (message == "blink") {
    Serial.print("Received message: ");
    Serial.println(message);

    digitalWrite(D1, HIGH);
    dc_trigger = true;
    encoder = 0;
    Serial.print("Relay: On");
  }
}

void reconnect() {
  while (!client.connected()) {
    Serial.println(WiFi.localIP());
    Serial.print("Attempting MQTT connection...");
    if (client.connect("ESP8266Client")) {
      Serial.println("connected");
      client.subscribe("esp32/led");
      Serial.println("Subscribed to esp32/led");
    } else {
      Serial.print("failed, rc=");
      Serial.print(client.state());
      Serial.println(" try again in 0.5 seconds");
      delay(500);
    }
  }
}

void setup() {
  pinMode(ledPin, OUTPUT);
  pinMode(D1, OUTPUT);
  Serial.begin(115200);
  setup_wifi();
  client.setServer(mqtt_server, 1883);
  client.setCallback(callback);
}

void loop() {
  if (!client.connected()) {
    reconnect();
  }
  client.loop();

  if(dc_trigger && encoder > 4000)
  {
    digitalWrite(D1, LOW);
    dc_trigger = false;
    Serial.println("Finish Pull Test");
  }

 if (client.connected()) {
    client.publish("esp8266", "Done");
    Serial.println("Published 'done' to esp8266");
  }
}
