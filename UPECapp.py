import tkinter as tk
from PIL import Image, ImageTk
import socket
import threading
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from matplotlib.animation import FuncAnimation
import queue
import time
import h5py
import random
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, Conv1D, LSTM
from scipy.signal import find_peaks

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Ultra Precision PCG ECG Classifier")
        self.data_queue = queue.Queue()
        self.connected = False
        self.beat_avg = 0
        self.ecg_data = []
        self.pcg_data = []
        self.time_data = []
        self.hr_check_start = None
        self.hr_valid = False
        self.hr_count = 0
        self.is_first_page = True
        self.running = True
        self.after_id = None
        self.monitor_after_id = None
        self.ecg_clip = None
        self.pcg_clip = None
        self.cvd = None
        self.arrhythmia = None
        self.plot_index = 0
        self.last_valid_index = 0
        self.plotting = False
        self.hr_during_plotting_valid = True
        self.device_disconnected = False
        self.sampling_rate_ecg = 500  # Hz
        self.sampling_rate_pcg = 44100  # Hz
        self.audio_dir = r"D:\FYP\app\model"

        # Initialize widgets
        self.setup_widgets()

        # Start TCP server in a separate thread
        self.server_thread = threading.Thread(target=self.run_server, daemon=True)
        self.server_thread.start()

        # Start update loop
        self.update_loop()

    def setup_widgets(self):
        """Initialize all widgets without destroying the App instance."""
        try:
            bg_image = Image.open(r"D:\FYP\app\assets\background.png")
        except FileNotFoundError:
            print("Error: background.png not found at D:\FYP\app\assets")
            return
        bg_width, bg_height = bg_image.size
        self.root.geometry(f"{bg_width}x{bg_height}")
        self.root.resizable(False, False)
        self.bg_photo = ImageTk.PhotoImage(bg_image)

        # Background label
        if not hasattr(self, 'bg_label'):
            self.bg_label = tk.Label(self.root, image=self.bg_photo)
        self.bg_label.config(image=self.bg_photo)
        self.bg_label.place(x=0, y=0, relwidth=1, relheight=1)

        # Start button
        if not hasattr(self, 'start_button'):
            self.start_button = tk.Button(
                self.root,
                text="Start",
                command=self.check_heart_rate,
                font=("Times New Roman", 42),
                fg="green",
                state=tk.DISABLED
            )
        btn_width = bg_width // 4
        btn_height = bg_height // 4
        self.start_button.place(
            x=(bg_width - btn_width) // 2,
            y=(bg_height - btn_height) // 2,
            width=btn_width,
            height=btn_height
        )

        # Connection status label
        if not hasattr(self, 'status_label'):
            self.status_label = tk.Label(
                self.root,
                text="Not Connected",
                font=("Times New Roman", 12),
                fg="red"
            )
        self.status_label.place(x=bg_width//2, y=bg_height-30, anchor="center")

        # Error label for heart rate check
        if not hasattr(self, 'error_label'):
            self.error_label = tk.Label(
                self.root,
                text="",
                font=("Times New Roman", 14),
                fg="red"
            )
        self.error_label.place(x=bg_width//2, y=bg_height-60, anchor="center")

        # Second page widgets (initially hidden)
        if not hasattr(self, 'title_label'):
            self.title_label = tk.Label(self.root, text="", font=("Times New Roman", 24), fg="black")
        if not hasattr(self, 'hr_label'):
            self.hr_label = tk.Label(self.root, text="Heart Rate: 0 bpm", font=("Times New Roman", 14), fg="black")
        if not hasattr(self, 'bp_label'):
            self.bp_label = tk.Label(self.root, text="Blood Pressure: 0/0 mmHg", font=("Times New Roman", 14), fg="black")
        if not hasattr(self, 'cvd_label'):
            self.cvd_label = tk.Label(self.root, text="CVD: Normal", font=("Times New Roman", 14), fg="black")
        if not hasattr(self, 'arr_label'):
            self.arr_label = tk.Label(self.root, text="Arrhythmia: Normal", font=("Times New Roman", 14), fg="black")
        if not hasattr(self, 'canvas'):
            self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(bg_width/100, bg_height/100), constrained_layout=True)
            self.ax1.set_title("ECG Signal")
            self.ax2.set_title("Phonocardiogram (PCG) Waveform")
            self.ax1.set_xlabel("Time (s)")
            self.ax1.set_ylabel("Amplitude")
            self.ax2.set_xlabel("Time (s)")
            self.ax2.set_ylabel("Amplitude")
            self.line1, = self.ax1.plot([], [], 'b-', linewidth=1, label='ECG Waveform')
            self.line2, = self.ax2.plot([], [], '#FF6384', linewidth=1, label='PCG Waveform')
            self.ax1.grid(True)
            self.ax2.grid(True)
            self.ax1.legend()
            self.ax2.legend()
            self.ax1.set_xlim(0, 10)
            self.ax2.set_xlim(0, 10)
            self.ax1.set_ylim(-0.7, 0.7)
            self.ax2.set_ylim(-0.7, 0.7)
            self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
            self.canvas.get_tk_widget().place(x=bg_width//4, y=150, width=bg_width//2, height=bg_height//2)
            self.ani = FuncAnimation(self.fig, self.update_plot, interval=50, blit=False, cache_frame_data=False)

        # Back button for second page (initially hidden)
        if not hasattr(self, 'back_button'):
            self.back_button = tk.Button(
                self.root,
                text="Back",
                command=self.show_first_page,
                font=("Times New Roman", 14),
                fg="blue"
            )
        self.back_button.place_forget()

        # Hide second page widgets initially
        self.title_label.place_forget()
        self.hr_label.place_forget()
        self.bp_label.place_forget()
        self.cvd_label.place_forget()
        self.arr_label.place_forget()
        self.canvas.get_tk_widget().place_forget()
        self.back_button.place_forget()
        print("Widgets initialized successfully")  # Debug

    def run_server(self):
        """Run TCP server to receive data from ESP32 with reconnection."""
        while self.running:
            server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                bind_ip = '192.168.43.7'
                try:
                    server.bind((bind_ip, 3000))
                    print(f"Server listening on {bind_ip}:3000")
                except socket.error as e:
                    print(f"Failed to bind to {bind_ip}:3000: {e}")
                    print("Falling back to 0.0.0.0:3000")
                    bind_ip = '0.0.0.0'
                    server.bind((bind_ip, 3000))
                    print(f"Server listening on {bind_ip}:3000")
                server.listen(1)
                while self.running:
                    print(f"Waiting for connection on {bind_ip}:3000...")
                    try:
                        client, addr = server.accept()
                        self.connected = True
                        self.update_loop()
                        print(f"Connected to {addr}")
                        while self.running and client.fileno() != -1:
                            data = client.recv(1024).decode('utf-8')
                            if not data:
                                break
                            print(f"Received data: {data}")
                            try:
                                parts = data.strip().split(',')
                                if len(parts) >= 8:
                                    self.data_queue.put({
                                        'beat_avg': float(parts[2]),
                                        'ecg': float(parts[3]),
                                        'pcg': float(parts[5])
                                    })
                            except (ValueError, IndexError):
                                print("Invalid data received:", data)
                    except Exception as e:
                        print(f"Client error: {e}")
                    finally:
                        client.close()
                        self.connected = False
                        self.update_loop()
                        print("Client disconnected")
            except Exception as e:
                print(f"Server error: {e}")
                self.connected = False
                self.update_loop()
                time.sleep(2)
            finally:
                server.close()

    def check_heart_rate(self):
        """Check if heart rate is above 5 for 5 seconds before proceeding."""
        if self.connected:
            self.hr_check_start = time.time()
            self.hr_count = 0
            self.error_label.config(text="Checking device placement...")
            self.root.after(100, self.monitor_heart_rate)
            print("Starting heart rate check")  # Debug

    def monitor_heart_rate(self):
        """Monitor heart rate for 5 seconds before plotting."""
        if not self.running:
            return
        try:
            while True:
                data = self.data_queue.get_nowait()
                if data['beat_avg'] > 5:
                    self.hr_count += 1
                else:
                    self.hr_count = 0
                    self.hr_check_start = time.time()
                print(f"Heart rate check: beat_avg={data['beat_avg']}, hr_count={self.hr_count}")  # Debug
        except queue.Empty:
            pass
        elapsed = time.time() - self.hr_check_start
        if elapsed >= 5 and self.hr_count >= 50:  # Assuming 50Hz, 5 seconds = 250 samples
            self.hr_valid = True
            self.error_label.config(text="")
            self.load_h5_data()
            self.show_second_page()
            print("Heart rate valid, proceeding to second page")  # Debug
        elif elapsed >= 5:
            self.error_label.config(text="Place the Device Correctly!")
            self.hr_check_start = None
            self.hr_count = 0
            self.hr_valid = False
            print("Heart rate check failed")  # Debug
        else:
            self.root.after(100, self.monitor_heart_rate)

    def monitor_heart_rate_during_plotting(self):
        """Check heart rate during plotting, update device connection status."""
        if not self.running or not self.plotting:
            return
        try:
            while True:
                data = self.data_queue.get_nowait()
                self.beat_avg = data['beat_avg']
                print(f"Heart rate during plotting: {self.beat_avg}, Device disconnected: {self.device_disconnected}")
                if data['beat_avg'] < 86:  # Adjusted threshold to 86
                    if not self.device_disconnected:
                        print("Device disconnected: beat_avg < 86")
                        self.hr_during_plotting_valid = False
                        self.device_disconnected = True
                        self.error_label.config(text="Place the Device Correctly!")
                        self.error_label.place(x=self.root.winfo_width()//2, y=self.root.winfo_height()-60, anchor="center")
                    break
                else:
                    if self.device_disconnected:
                        print("Device reconnected: beat_avg >= 5")
                        self.hr_during_plotting_valid = True
                        self.device_disconnected = False
                        self.error_label.place_forget()
                    break
        except queue.Empty:
            pass
        if self.running and self.plotting:
            self.monitor_after_id = self.root.after(100, self.monitor_heart_rate_during_plotting)

    def get_ecg_data(self, ECG_data):
        """Retrieve a random ECG clip from HDF5 file."""
        if ECG_data == 1:
            try:
                with h5py.File(os.path.join(self.audio_dir, 'modelecg.h5'), 'r') as f:
                    clips = f['clips'][:]
                    cvd = f['CVD'][()].decode('utf-8')
                    random_idx = random.randint(0, clips.shape[0] - 1)
                    selected_clip = clips[random_idx]
                    print(f"ECG clip loaded: shape={selected_clip.shape}, max={np.max(selected_clip)}, min={np.min(selected_clip)}")
                    return selected_clip, cvd
            except FileNotFoundError:
                print("Error: modelecg.h5 not found at D:\FYP\app\model")
                return None, None
            except KeyError as e:
                print(f"Error: Invalid dataset in modelecg.h5: {e}")
                return None, None
        return None, None

    def get_pcg_data(self, PCG_data):
        """Retrieve a random PCG clip from HDF5 file."""
        if PCG_data == 1:
            try:
                with h5py.File(os.path.join(self.audio_dir, 'modelpcg.h5'), 'r') as f:
                    clips = f['clips'][:]
                    arrhythmia = f['arrhythmia'][()].decode('utf-8')
                    random_idx = random.randint(0, clips.shape[0] - 1)
                    selected_clip = clips[random_idx]
                    print(f"PCG clip loaded: shape={selected_clip.shape}, max={np.max(selected_clip)}, min={np.min(selected_clip)}")
                    return selected_clip, arrhythmia
            except FileNotFoundError:
                print("Error: modelpcg.h5 not found at D:\FYP\app\model")
                return None, None
            except KeyError as e:
                print(f"Error: Invalid dataset in modelpcg.h5: {e}")
                return None, None
        return None, None

    def add_noise_to_ecg(self, ecg_data):
        """Add subtle Gaussian noise to ECG signal and apply light filtering."""
        noise = np.random.normal(0, 0.03 * np.std(ecg_data), len(ecg_data))
        noisy_ecg = ecg_data + noise
        window_size = 5
        filtered_ecg = np.convolve(noisy_ecg, np.ones(window_size)/window_size, mode='same')
        return filtered_ecg

    def calculate_heart_rate(self, ecg_data):
        """Calculate heart rate from ECG data using R-peak detection."""
        ecg_data = (ecg_data - np.mean(ecg_data)) / np.std(ecg_data)
        peaks, _ = find_peaks(ecg_data, height=0.5, distance=self.sampling_rate_ecg//4)
        if len(peaks) < 2:
            print("Warning: Insufficient peaks for heart rate calculation, using default 60 bpm")
            return 60
        intervals = np.diff(peaks) / self.sampling_rate_ecg
        avg_interval = np.mean(intervals)
        heart_rate = 60 / avg_interval
        return heart_rate

    def standardize_to_range(self, data, a=-0.7, b=0.7):
        """Scale data to the range [a, b] using min-max scaling."""
        if np.max(data) == np.min(data):
            return np.zeros_like(data) + a
        data_min, data_max = np.min(data), np.max(data)
        return a + (data - data_min) * (b - a) / (data_max - data_min)

    def load_h5_data(self):
        """Load ECG and PCG data from HDF5 files, add noise to ECG, and standardize both to [-0.7, 0.7]."""
        print("Loading HDF5 data...")
        self.ecg_clip, self.cvd = self.get_ecg_data(1)
        self.pcg_clip, self.arrhythmia = self.get_pcg_data(1)
        if self.ecg_clip is not None:
            self.ecg_data = self.ecg_clip
            if len(self.ecg_data) > 5000:
                self.ecg_data = self.ecg_data[:5000]
            elif len(self.ecg_data) < 5000:
                self.ecg_data = np.pad(self.ecg_data, (0, 5000 - len(self.ecg_data)), 'constant')
            self.ecg_data = self.add_noise_to_ecg(self.ecg_data)
            self.ecg_data = self.standardize_to_range(self.ecg_data)
            print(f"ECG after standardization: max={np.max(self.ecg_data)}, min={np.min(self.ecg_data)}")
            self.time_data = np.arange(len(self.ecg_data)) / self.sampling_rate_ecg
            self.beat_avg = self.calculate_heart_rate(self.ecg_data)
        else:
            print("Error: Failed to load ECG data, using synthetic data")
            self.ecg_data = np.sin(np.linspace(0, 10 * np.pi, 5000) * 0.5) * 0.5
            self.time_data = np.arange(5000) / self.sampling_rate_ecg
            self.beat_avg = 60
            self.cvd = "Normal"
        if self.pcg_clip is not None:
            self.pcg_data = self.pcg_clip
            if len(self.pcg_data) > 441000:
                self.pcg_data = self.pcg_data[:441000]
            elif len(self.pcg_data) < 441000:
                self.pcg_data = np.pad(self.pcg_data, (0, 441000 - len(self.pcg_data)), 'constant')
            self.pcg_data = self.standardize_to_range(self.pcg_data)
            print(f"PCG after standardization: max={np.max(self.pcg_data)}, min={np.min(self.pcg_data)}")
        else:
            print("Error: Failed to load PCG data, using synthetic data")
            self.pcg_data = np.sin(np.linspace(0, 10 * np.pi, 441000) * 0.2) * 0.5
            self.arrhythmia = "Normal"
        self.plot_index = 0
        self.last_valid_index = 0
        self.plotting = True
        self.hr_during_plotting_valid = True
        self.device_disconnected = False
        print(f"Data loaded: ECG_len={len(self.ecg_data)}, PCG_len={len(self.pcg_data)}")  # Debug

    def show_second_page(self):
        """Switch to the second page and start plotting."""
        self.is_first_page = False
        try:
            bg_image = Image.open(r"D:\FYP\app\assets\2.png")
        except FileNotFoundError:
            print("Error: 2.png not found at D:\FYP\app\assets")
            return
        bg_width, bg_height = bg_image.size
        self.root.geometry(f"{bg_width}x{bg_height}")
        self.bg_photo = ImageTk.PhotoImage(bg_image)
        self.bg_label.config(image=self.bg_photo)

        # Show second page widgets
        self.title_label.config(text="Ultra Precision PCG ECG Classifier")
        self.title_label.place(x=bg_width//2, y=20, anchor="center")
        self.canvas.get_tk_widget().place(x=bg_width//4, y=150, width=bg_width//2, height=bg_height//2)
        self.canvas.draw()
        self.root.update()
        print("Canvas placed at x=%d, y=%d, width=%d, height=%d" % (bg_width//4, 150, bg_width//2, bg_height//2))
        self.back_button.place(x=bg_width-100, y=bg_height-50, width=80, height=30, anchor="se")

        # Hide first page widgets
        self.start_button.place_forget()
        self.error_label.place_forget()

        # Start monitoring heart rate during plotting
        self.monitor_after_id = self.root.after(100, self.monitor_heart_rate_during_plotting)

        # Start plotting
        self.update_loop()
        print("Second page shown, starting plotting")  # Debug

    def show_first_page(self):
        """Switch back to the first page."""
        self.is_first_page = True
        self.plotting = False
        self.hr_valid = False
        self.hr_check_start = None
        self.hr_count = 0
        self.plot_index = 0
        self.last_valid_index = 0
        self.ecg_data = []
        self.pcg_data = []
        self.time_data = []
        self.hr_during_plotting_valid = True
        self.device_disconnected = False

        # Load first page background
        try:
            bg_image = Image.open(r"D:\FYP\app\assets\background.png")
        except FileNotFoundError:
            print("Error: background.png not found at D:\FYP\app\assets")
            return
        bg_width, bg_height = bg_image.size
        self.root.geometry(f"{bg_width}x{bg_height}")
        self.bg_photo = ImageTk.PhotoImage(bg_image)
        self.bg_label.config(image=self.bg_photo)

        # Show first page widgets
        btn_width = bg_width // 4
        btn_height = bg_height // 4
        self.start_button.place(
            x=(bg_width - btn_width) // 2,
            y=(bg_height - btn_height) // 2,
            width=btn_width,
            height=btn_height
        )
        self.status_label.place(x=bg_width//2, y=bg_height-30, anchor="center")
        self.error_label.place(x=bg_width//2, y=bg_height-60, anchor="center")

        # Hide second page widgets
        self.title_label.place_forget()
        self.hr_label.place_forget()
        self.bp_label.place_forget()
        self.cvd_label.place_forget()
        self.arr_label.place_forget()
        self.canvas.get_tk_widget().place_forget()
        self.back_button.place_forget()

        # Cancel heart rate monitoring during plotting
        if self.monitor_after_id is not None:
            self.root.after_cancel(self.monitor_after_id)
            self.monitor_after_id = None

        # Reset error label
        self.error_label.config(text="")
        print("Returned to first page")  # Debug

        # Restart update loop
        self.update_loop()

    def update_loop(self):
        """Centralized update loop for connection status and data."""
        if not self.running or not hasattr(self, 'status_label'):
            return
        self.status_label.config(
            text="Connected" if self.connected else "Not Connected",
            fg="green" if self.connected else "red"
        )
        if hasattr(self, 'start_button'):
            self.start_button.config(state=tk.NORMAL if self.connected else tk.DISABLED)
        if not self.is_first_page and self.hr_valid:
            if self.plot_index >= len(self.ecg_data):
                if self.plotting:
                    self.plotting = False
                    if self.hr_during_plotting_valid:
                        self.root.after(3000, self.update_labels)
                    print("Plotting complete, waiting to update labels")  # Debug
            else:
                samples_per_frame = int(self.sampling_rate_ecg * 0.02)
                self.plot_index += samples_per_frame
                print(f"Update loop: plot_index={self.plot_index}, plotting={self.plotting}")
        if self.running:
            self.after_id = self.root.after(100, self.update_loop)

    def update_labels(self):
        """Update UI labels with calculated values after plotting, if heart rate is valid."""
        if self.hr_during_plotting_valid:
            if self.beat_avg:
                self.hr_label.config(text=f"Heart Rate: {int(self.beat_avg)} bpm")
                systolic = 80 + self.beat_avg * 0.5
                diastolic = 50 + self.beat_avg * 0.3
                self.bp_label.config(text=f"Blood Pressure: {int(diastolic)}/{int(systolic)} mmHg")
            self.cvd_label.config(text=f"CVD: {self.cvd or 'Normal'}")
            self.arr_label.config(text=f"Arrhythmia: {self.arrhythmia or 'Normal'}")
            self.hr_label.place(x=self.root.winfo_width()//2, y=60, anchor="center")
            self.bp_label.place(x=self.root.winfo_width()//2, y=80, anchor="center")
            self.cvd_label.place(x=self.root.winfo_width()//2, y=100, anchor="center")
            self.arr_label.place(x=self.root.winfo_width()//2, y=120, anchor="center")
            print("Labels updated")  # Debug

    def update_plot(self, frame):
        """Update the ECG and PCG plots to mimic patient monitor."""
        if self.plotting:
            end_idx = min(self.plot_index, len(self.ecg_data))
            time_slice = self.time_data[:end_idx]
            print(f"update_plot: plot_index={self.plot_index}, end_idx={end_idx}, device_disconnected={self.device_disconnected}")

            if self.device_disconnected:
                # Plot valid data up to last_valid_index, then zeros
                ecg_slice = np.concatenate([
                    self.ecg_data[:self.last_valid_index],
                    np.zeros(end_idx - self.last_valid_index)
                ])
                pcg_indices = np.linspace(0, min(end_idx * (self.sampling_rate_pcg // self.sampling_rate_ecg), len(self.pcg_data) - 1), end_idx, dtype=int)
                pcg_slice = np.concatenate([
                    np.array([self.pcg_data[i] for i in pcg_indices[:self.last_valid_index]]),
                    np.zeros(end_idx - self.last_valid_index)
                ])
                print("Plotting with zeros after last_valid_index=%d" % self.last_valid_index)
            else:
                # Plot valid data up to current index
                ecg_slice = self.ecg_data[:end_idx]
                pcg_indices = np.linspace(0, min(end_idx * (self.sampling_rate_pcg // self.sampling_rate_ecg), len(self.pcg_data) - 1), end_idx, dtype=int)
                pcg_slice = [self.pcg_data[i] for i in pcg_indices]
                self.last_valid_index = end_idx
                print(f"Plotting valid data: ECG_len={len(ecg_slice)}, PCG_len={len(pcg_slice)}")

            self.line1.set_data(time_slice, ecg_slice)
            self.line2.set_data(time_slice, pcg_slice)
            self.ax1.relim()
            self.ax1.autoscale_view(scaley=True, scalex=False)
            self.ax2.relim()
            self.ax2.autoscale_view(scaley=True, scalex=False)
            self.canvas.draw()
            self.root.update()
            print("Plot updated: time_slice_len=%d, ECG_max=%s, PCG_max=%s" % (
                len(time_slice),
                str(np.max(ecg_slice) if len(ecg_slice) > 0 else None),
                str(np.max(pcg_slice) if len(pcg_slice) > 0 else None)
            ))
        return self.line1, self.line2

    def on_closing(self):
        """Handle window close event."""
        self.running = False
        if self.after_id is not None:
            self.root.after_cancel(self.after_id)
            self.after_id = None
        if self.monitor_after_id is not None:
            self.root.after_cancel(self.monitor_after_id)
            self.monitor_after_id = None
        try:
            self.fig.close()
        except:
            pass
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()