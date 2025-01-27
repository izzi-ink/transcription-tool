import warnings
import customtkinter as ctk
from tkinter import filedialog, messagebox
import threading
import os
import certifi
import whisper
from queue import Queue
import torch

# Filter out specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module="whisper.transcribe")
warnings.filterwarnings("ignore", category=FutureWarning, module="whisper")

# Ensure we're using the right device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class TranscriptionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Whisper Transcription Tool")
        self.root.geometry("900x700")
        
        # Initialize model to None
        self.whisper_model = None
        self.queue = Queue()
        
        # Set theme
        ctk.set_appearance_mode("light")
        ctk.set_default_color_theme("blue")
        
        # Create main frame with padding
        self.main_frame = ctk.CTkFrame(root)
        self.main_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Heading
        ctk.CTkLabel(
            self.main_frame,
            text="Whisper Transcription Tool",
            font=ctk.CTkFont(size=24, weight="bold")
        ).pack(pady=(0, 20))
        
        # File selection frame
        self.file_frame = ctk.CTkFrame(self.main_frame)
        self.file_frame.pack(fill="x", padx=10, pady=10)
        
        self.file_path = ctk.StringVar()
        self.file_entry = ctk.CTkEntry(
            self.file_frame, 
            textvariable=self.file_path,
            width=400,
            placeholder_text="Select an audio file..."
        )
        self.file_entry.pack(side="left", padx=(10, 10), fill="x", expand=True)
        
        self.browse_button = ctk.CTkButton(
            self.file_frame,
            text="Browse",
            command=self.browse_file,
            width=100
        )
        self.browse_button.pack(side="right", padx=10)
        
        # Settings frame
        self.settings_frame = ctk.CTkFrame(self.main_frame)
        self.settings_frame.pack(fill="x", padx=10, pady=10)
        
        # Model selection
        model_label = ctk.CTkLabel(self.settings_frame, text="Model:")
        model_label.pack(side="left", padx=10)
        
        self.model_var = ctk.StringVar(value="base")
        self.model_combo = ctk.CTkOptionMenu(
            self.settings_frame,
            variable=self.model_var,
            values=['tiny', 'base', 'small', 'medium', 'large']
        )
        self.model_combo.pack(side="left", padx=10)
        
        # Device info
        self.device_info = ctk.CTkLabel(
            self.settings_frame,
            text=f"Running on: {'CUDA' if torch.cuda.is_available() else 'CPU'}",
            font=ctk.CTkFont(weight="bold")
        )
        self.device_info.pack(side="right", padx=20)
        
        # Status and Progress
        self.status_var = ctk.StringVar(value="Ready")
        self.status_label = ctk.CTkLabel(
            self.main_frame,
            textvariable=self.status_var,
            font=ctk.CTkFont(size=12)
        )
        self.status_label.pack(pady=10)
        
        self.progress = ctk.CTkProgressBar(self.main_frame)
        self.progress.pack(fill="x", padx=10, pady=(0, 10))
        self.progress.set(0)
        
        # Transcription output
        self.output_frame = ctk.CTkFrame(self.main_frame)
        self.output_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        self.output_text = ctk.CTkTextbox(
            self.output_frame,
            wrap="word",
            font=ctk.CTkFont(size=12),
            height=300
        )
        self.output_text.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Control buttons frame
        self.button_frame = ctk.CTkFrame(self.main_frame)
        self.button_frame.pack(fill="x", padx=10, pady=10)
        
        self.transcribe_button = ctk.CTkButton(
            self.button_frame,
            text="Transcribe",
            command=self.start_transcription,
            font=ctk.CTkFont(weight="bold"),
            height=40
        )
        self.transcribe_button.pack(side="left", padx=10, pady=10)
        
        self.save_button = ctk.CTkButton(
            self.button_frame,
            text="Save Transcription",
            command=self.save_transcription,
            height=40
        )
        self.save_button.pack(side="right", padx=10, pady=10)
        
        # Start queue checking
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
    
    def check_queue(self):
        try:
            while True:
                msg = self.queue.get_nowait()
                
                if msg.startswith("status:"):
                    self.status_var.set(msg[7:])
                elif msg.startswith("progress:"):
                    self.progress.set(float(msg[9:]))
                elif msg.startswith("complete:"):
                    self.output_text.delete("0.0", "end")
                    self.output_text.insert("end", msg[9:])
                    self.transcribe_button.configure(state="normal")
                    self.status_var.set("Transcription complete!")
                elif msg.startswith("error:"):
                    messagebox.showerror("Error", msg[6:])
                    self.transcribe_button.configure(state="normal")
                    self.status_var.set("Ready")
        except:
            pass
        finally:
            self.root.after(100, self.check_queue)
    
    def start_transcription(self):
        if not self.file_path.get():
            messagebox.showwarning("Warning", "Please select an audio file first.")
            return
        
        self.transcribe_button.configure(state="disabled")
        self.status_var.set("Starting transcription...")
        self.progress.set(0)
        
        thread = threading.Thread(target=self.transcribe)
        thread.daemon = True
        thread.start()
    
    def transcribe(self):
        try:
            # Load model if not already loaded
            if self.whisper_model is None:
                self.queue.put("status:Loading Whisper model (this might take a few minutes)...")
                self.queue.put("progress:10")
                
                try:
                    self.whisper_model = whisper.load_model(
                        self.model_var.get(),
                        device=DEVICE
                    )
                    self.queue.put("status:Model loaded successfully!")
                    self.queue.put("progress:30")
                except Exception as e:
                    self.queue.put(f"error:Failed to load model: {str(e)}")
                    return
            
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
    
    def save_transcription(self):
        if not self.output_text.get("0.0", "end").strip():
            messagebox.showwarning("Warning", "No transcription to save.")
            return
            
        filename = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")]
        )
        if filename:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(self.output_text.get("0.0", "end"))
            messagebox.showinfo("Success", "Transcription saved successfully!")

if __name__ == "__main__":
    root = ctk.CTk()
    app = TranscriptionApp(root)
    root.mainloop()