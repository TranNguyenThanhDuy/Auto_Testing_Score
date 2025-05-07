import paho.mqtt.client as mqtt

broker = "192.168.186.210"
topic = "esp32/led"
message = "blink"

client = mqtt.Client()
try:
    client.connect(broker, 1883, 60)
    print("✅ Connected!")
    client.publish(topic, message)
    client.disconnect()
    print("Sent:", message)
except Exception as e:
    print("❌ Connection failed:", e)

