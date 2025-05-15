# import paho.mqtt.client as mqtt
# import subprocess
#
# def on_message(client, userdata, msg):
#     payload = msg.payload.decode().strip().lower()
#     print(f"📩 Nhận tin nhắn trên topic {msg.topic}: {payload}")
#     if payload == 'score':
#         print("▶️ Thực thi chấm điểm...")
#         subprocess.run(["python", "chamthi.py"])
#         print("✅ Xử lý xong!")
#
# client = mqtt.Client()
# client.connect("192.168.215.209", 1883, 60)  # Thay bằng IP broker nếu dùng máy khác
# client.subscribe("test/command")
# client.on_message = on_message
#
# print("🕹️ Đang lắng nghe tín hiệu 'score' trên topic 'image/command'...")
# client.loop_forever()

import tkinter as tk
from tkinter import filedialog, messagebox
import paho.mqtt.client as mqtt
import subprocess
import threading
import os

# === Hàm xử lý MQTT ===
def on_message(client, userdata, msg):
    payload = msg.payload.decode().strip().lower()
    print(f"📩 Nhận tin nhắn trên topic {msg.topic}: {payload}")

    if payload == 'score':
        print("▶️ Đang chấm điểm...")
        try:
            subprocess.run([
                "python", "chamthi.py",
                "--answers", answers_path.get(),
                "--output", output_csv.get()
            ])
            print("✅ Chấm điểm xong!")
        except Exception as e:
            messagebox.showerror("Lỗi khi chạy chamthi.py", str(e))

    # Nếu payload là kết quả điểm dạng sbd:diem
    elif ":" in payload:
        try:
            sbd, diem = payload.split(":")
            result_box.config(state='normal')  # Cho phép ghi
            result_box.insert(tk.END, f"SBD: {sbd.strip()} | Điểm: {diem.strip()}\n")
            result_box.see(tk.END)  # Tự động scroll xuống dòng cuối
            result_box.config(state='disabled')  # Khóa lại
        except Exception as e:
            print("❌ Không thể phân tích kết quả:", e)

def start_mqtt(broker_ip):
    try:
        client =  mqtt.Client(client_id="", userdata=None, protocol=mqtt.MQTTv311, transport="tcp")
        client.connect(broker_ip, 1883, 60)
        client.subscribe("esp8266/score")
        client.subscribe("Score/finish")
        client.on_message = on_message
        print(f"🕹️ Lắng nghe từ broker {broker_ip} trên topic 'esp8266/score' và 'Score/finish'")
        client.loop_forever()
    except Exception as e:
        messagebox.showerror("Không kết nối được", str(e))

def run_mqtt_thread():
    broker = broker_ip.get().strip()
    if not broker:
        messagebox.showwarning("Thiếu IP", "Vui lòng nhập IP broker!")
        return
    if not os.path.isfile(answers_path.get()):
        messagebox.showwarning("Thiếu file", "Vui lòng chọn file answers.json!")
        return
    if not output_csv.get():
        messagebox.showwarning("Thiếu tên CSV", "Vui lòng nhập tên file CSV!")
        return

    threading.Thread(target=start_mqtt, args=(broker,), daemon=True).start()

# === Hàm chọn file answers.json ===
def browse_answers():
    filename = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")])
    if filename:
        answers_path.set(filename)

# === Giao diện UI ===
root = tk.Tk()
root.title("Chấm điểm MQTT GUI")

# Khai báo biến
broker_ip = tk.StringVar()
answers_path = tk.StringVar()
output_csv = tk.StringVar(value="grading_result.csv")

tk.Label(root, text="Nhập IP Broker:").grid(row=0, column=0, sticky='e', padx=5, pady=5)
tk.Entry(root, textvariable=broker_ip, width=30).grid(row=0, column=1, padx=5, pady=5)

tk.Label(root, text="Chọn File đáp án:").grid(row=1, column=0, sticky='e', padx=5, pady=5)
tk.Entry(root, textvariable=answers_path, width=30).grid(row=1, column=1, padx=5, pady=5)
tk.Button(root, text="Chọn...", command=browse_answers).grid(row=1, column=2, padx=5, pady=5)

tk.Label(root, text="Tên file CSV lưu kết quả:").grid(row=2, column=0, sticky='e', padx=5, pady=5)
tk.Entry(root, textvariable=output_csv, width=30).grid(row=2, column=1, padx=5, pady=5)

tk.Button(root, text="Start", command=run_mqtt_thread).grid(row=3, column=0, columnspan=3, pady=10)

# === Khung hiển thị danh sách kết quả ===
result_box = tk.Text(root, height=10, width=50, font=("Courier", 12))
result_box.grid(row=4, column=0, columnspan=3, pady=15)
result_box.insert(tk.END, "📋 Danh sách kết quả chấm bài:\n")
result_box.config(state='disabled')

root.mainloop()

