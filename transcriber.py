import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import os
import certifi
import whisper
from queue import Queue
import torch

class TranscriptionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Whisper Transcription Tool")
        self.root.geometry("800x600")
        
        # Set SSL Certificate
        os.environ["SSL_CERT_FILE"] = certifi.where()
        
        # Message queue for thread communication
        self.queue = Queue()
        self.root.after(100, self.check_queue)
        
        # Create main frame
        self.main_frame = ttk.Frame(root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # File selection
        self.file_frame = ttk.LabelFrame(self.main_frame, text="Audio File", padding="5")
        self.file_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        self.file_path = tk.StringVar()
        self.file_entry = ttk.Entry(self.file_frame, textvariable=self.file_path, width=50)
        self.file_entry.grid(row=0, column=0, padx=5)
        
        self.browse_button = ttk.Button(self.file_frame, text="Browse", command=self.browse_file)
        self.browse_button.grid(row=0, column=1, padx=5)
        
        # Model selection
        self.model_frame = ttk.LabelFrame(self.main_frame, text="Settings", padding="5")
        self.model_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Label(self.model_frame, text="Model:").grid(row=0, column=0, padx=5)
        self.model_var = tk.StringVar(value="base")  # Changed default to base
        self.model_combo = ttk.Combobox(self.model_frame, textvariable=self.model_var)
        self.model_combo['values'] = ('tiny', 'base', 'small', 'medium', 'large')
        self.model_combo.grid(row=0, column=1, padx=5)
        
        # Device info
        self.device_info = ttk.Label(self.model_frame, 
                                   text=f"Running on: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
        self.device_info.grid(row=0, column=2, padx=5)
        
        # Status label
        self.status_var = tk.StringVar(value="Ready")
        self.status_label = ttk.Label(self.main_frame, textvariable=self.status_var)
        self.status_label.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress = ttk.Progressbar(self.main_frame, variable=self.progress_var, maximum=100)
        self.progress.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        # Transcription output
        self.output_frame = ttk.LabelFrame(self.main_frame, text="Transcription", padding="5")
        self.output_frame.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        
        self.output_text = tk.Text(self.output_frame, wrap=tk.WORD, width=80, height=20)
        self.output_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Scrollbar for output
        self.scrollbar = ttk.Scrollbar(self.output_frame, orient=tk.VERTICAL, command=self.output_text.yview)
        self.scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.output_text['yscrollcommand'] = self.scrollbar.set
        
        # Control buttons
        self.button_frame = ttk.Frame(self.main_frame)
        self.button_frame.grid(row=5, column=0, columnspan=2, pady=5)
        
        self.transcribe_button = ttk.Button(self.button_frame, text="Transcribe", command=self.start_transcription)
        self.transcribe_button.grid(row=0, column=0, padx=5)
        
        self.save_button = ttk.Button(self.button_frame, text="Save Transcription", command=self.save_transcription)
        self.save_button.grid(row=0, column=1, padx=5)
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        self.main_frame.columnconfigure(1, weight=1)
        self.main_frame.rowconfigure(4, weight=1)
        self.output_frame.columnconfigure(0, weight=1)
        self.output_frame.rowconfigure(0, weight=1)

        # Initialize model to None
        self.whisper_model = None

    def check_queue(self):
        """Check for messages from the worker thread."""
        try:
            while True:
                msg = self.queue.get_nowait()
                
                if msg.startswith("status:"):
                    self.status_var.set(msg[7:])
                elif msg.startswith("progress:"):
                    self.progress_var.set(float(msg[9:]))
                elif msg.startswith("complete:"):
                    self.output_text.delete(1.0, tk.END)
                    self.output_text.insert(tk.END, msg[9:])
                    self.transcribe_button.config(state=tk.NORMAL)
                    self.status_var.set("Transcription complete!")
                elif msg.startswith("error:"):
                    messagebox.showerror("Error", msg[6:])
                    self.transcribe_button.config(state=tk.NORMAL)
                    self.status_var.set("Ready")
                
        except Exception:
            pass
        finally:
            # Schedule the next queue check
            self.root.after(100, self.check_queue)

    def browse_file(self):
        filename = filedialog.askopenfilename(
            filetypes=[
                ("Audio Files", "*.mp3 *.wav *.m4a *.ogg"),
                ("All Files", "*.*")
            ]
        )
        if filename:
            self.file_path.set(filename)

    def transcribe(self):
        try:
            # Load model if not already loaded
            if self.whisper_model is None:
                self.queue.put("status:Loading Whisper model (this might take a few minutes)...")
                self.queue.put("progress:10")
                
                # Determine device
                device = "cuda" if torch.cuda.is_available() else "cpu"
                self.queue.put(f"status:Loading model on {device.upper()}...")
                
                self.whisper_model = whisper.load_model(
                    self.model_var.get(),
                    device=device
                )
                self.queue.put("status:Model loaded successfully!")
                self.queue.put("progress:30")
            
            self.queue.put("status:Processing audio file...")
            self.queue.put("progress:40")
            
            # Transcribe
            self.queue.put("status:Transcribing audio (this might take a while)...")
            self.queue.put("progress:50")
            result = self.whisper_model.transcribe(self.file_path.get())
            
            self.queue.put("status:Finalizing transcription...")
            self.queue.put("progress:90")
            self.queue.put(f"complete:{result['text']}")
            self.queue.put("progress:100")
            
        except Exception as e:
            self.queue.put(f"error:Transcription failed: {str(e)}")
            self.whisper_model = None  # Reset model on error

    def start_transcription(self):
        if not self.file_path.get():
            messagebox.showwarning("Warning", "Please select an audio file first.")
            return
        
        self.transcribe_button.config(state=tk.DISABLED)
        self.status_var.set("Starting transcription...")
        self.progress_var.set(0)
        
        thread = threading.Thread(target=self.transcribe)
        thread.daemon = True
        thread.start()

    def save_transcription(self):
        if not self.output_text.get(1.0, tk.END).strip():
            messagebox.showwarning("Warning", "No transcription to save.")
            return
            
        filename = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")]
        )
        if filename:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(self.output_text.get(1.0, tk.END))
            messagebox.showinfo("Success", "Transcription saved successfully!")

if __name__ == "__main__":
    root = tk.Tk()
    app = TranscriptionApp(root)
    root.mainloop()