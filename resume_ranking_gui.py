import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import json
import os
from sentence_transformers import SentenceTransformer
from sentence_transformers import util
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity
import spacy
import fitz
from collections import defaultdict

class ResumeRankingGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Resume Ranking System")
        self.root.geometry("1200x800")
        
        # Initialize models
        self.model = SentenceTransformer('all-mpnet-base-v2')
        self.nlp = spacy.load("en_core_web_lg")
        
        # Load existing data
        self.load_data()
        
        # Create main container
        self.main_container = ttk.Frame(root, padding="10")
        self.main_container.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Create tabs
        self.tab_control = ttk.Notebook(self.main_container)
        
        # Add Candidate Tab
        self.add_candidate_tab = ttk.Frame(self.tab_control)
        self.tab_control.add(self.add_candidate_tab, text='Add Candidate')
        
        # Rankings Tab
        self.rankings_tab = ttk.Frame(self.tab_control)
        self.tab_control.add(self.rankings_tab, text='Rankings')
        
        # Job Description Tab
        self.job_desc_tab = ttk.Frame(self.tab_control)
        self.tab_control.add(self.job_desc_tab, text='Job Description')
        
        self.tab_control.pack(expand=1, fill="both")
        
        # Setup tabs
        self.setup_add_candidate_tab()
        self.setup_rankings_tab()
        self.setup_job_desc_tab()
        
    def load_data(self):
        try:
            with open('hybrid_matching_results.json', 'r') as f:
                self.rankings_data = json.load(f)
        except FileNotFoundError:
            self.rankings_data = []
            
    def setup_add_candidate_tab(self):
        # Create form frame
        form_frame = ttk.LabelFrame(self.add_candidate_tab, text="Candidate Information", padding="10")
        form_frame.grid(row=0, column=0, padx=10, pady=5, sticky="nsew")
        
        # Name field
        ttk.Label(form_frame, text="Full Name:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.name_var = tk.StringVar()
        ttk.Entry(form_frame, textvariable=self.name_var, width=40).grid(row=0, column=1, sticky=tk.W, pady=5)
        
        # Resume file selection
        ttk.Label(form_frame, text="Resume PDF:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.resume_path_var = tk.StringVar()
        ttk.Entry(form_frame, textvariable=self.resume_path_var, width=40).grid(row=1, column=1, sticky=tk.W, pady=5)
        ttk.Button(form_frame, text="Browse", command=self.browse_resume).grid(row=1, column=2, padx=5, pady=5)
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(form_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.grid(row=2, column=0, columnspan=3, sticky="ew", padx=5, pady=5)
        
        # Submit button
        ttk.Button(form_frame, text="Add Candidate", command=self.add_candidate).grid(row=3, column=0, columnspan=3, pady=20)
        
    def setup_rankings_tab(self):
        # Create rankings frame
        rankings_frame = ttk.Frame(self.rankings_tab, padding="10")
        rankings_frame.grid(row=0, column=0, sticky="nsew")
        
        # Create Treeview
        self.tree = ttk.Treeview(rankings_frame, columns=('Rank', 'Name', 'Score', 'Transformer', 'TF-IDF', 'Section'), show='headings')
        
        # Define headings
        self.tree.heading('Rank', text='Rank')
        self.tree.heading('Name', text='Name')
        self.tree.heading('Score', text='Combined Score')
        self.tree.heading('Transformer', text='Transformer Score')
        self.tree.heading('TF-IDF', text='TF-IDF Score')
        self.tree.heading('Section', text='Section Score')
        
        # Define columns
        self.tree.column('Rank', width=50)
        self.tree.column('Name', width=200)
        self.tree.column('Score', width=100)
        self.tree.column('Transformer', width=100)
        self.tree.column('TF-IDF', width=100)
        self.tree.column('Section', width=100)
        
        # Add scrollbar
        y_scrollbar = ttk.Scrollbar(rankings_frame, orient=tk.VERTICAL, command=self.tree.yview)
        x_scrollbar = ttk.Scrollbar(rankings_frame, orient=tk.HORIZONTAL, command=self.tree.xview)
        self.tree.configure(yscrollcommand=y_scrollbar.set, xscrollcommand=x_scrollbar.set)
        
        # Grid layout
        self.tree.grid(row=0, column=0, sticky="nsew")
        y_scrollbar.grid(row=0, column=1, sticky="ns")
        x_scrollbar.grid(row=1, column=0, sticky="ew")
        
        # Configure grid weights
        rankings_frame.grid_columnconfigure(0, weight=1)
        rankings_frame.grid_rowconfigure(0, weight=1)
        
        # Add refresh button
        ttk.Button(rankings_frame, text="Refresh Rankings", command=self.update_rankings_display).grid(row=2, column=0, pady=10)
        
        # Update rankings display
        self.update_rankings_display()
        
    def setup_job_desc_tab(self):
        # Create job description frame
        job_frame = ttk.Frame(self.job_desc_tab, padding="10")
        job_frame.grid(row=0, column=0, sticky="nsew")
        
        # Add job description text area
        ttk.Label(job_frame, text="Job Description:").grid(row=0, column=0, sticky=tk.W, pady=5)
        
        self.job_desc_text = tk.Text(job_frame, wrap=tk.WORD, width=60, height=20)
        self.job_desc_text.grid(row=1, column=0, sticky="nsew", pady=10)
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(job_frame, orient=tk.VERTICAL, command=self.job_desc_text.yview)
        self.job_desc_text.configure(yscrollcommand=scrollbar.set)
        scrollbar.grid(row=1, column=1, sticky="ns")
        
        # Add save button
        ttk.Button(job_frame, text="Save Job Description", command=self.save_job_description).grid(row=2, column=0, pady=10)
        
        # Load existing job description if available
        try:
            with open('job_description.txt', 'r') as f:
                self.job_desc_text.insert('1.0', f.read())
        except FileNotFoundError:
            pass
            
    def browse_resume(self):
        filename = filedialog.askopenfilename(
            title="Select Resume PDF",
            filetypes=(("PDF files", "*.pdf"), ("All files", "*.*"))
        )
        if filename:
            self.resume_path_var.set(filename)
            
    def save_job_description(self):
        job_desc = self.job_desc_text.get('1.0', tk.END).strip()
        with open('job_description.txt', 'w') as f:
            f.write(job_desc)
        messagebox.showinfo("Success", "Job description saved successfully!")
            
    def add_candidate(self):
        name = self.name_var.get().strip()
        resume_path = self.resume_path_var.get().strip()
        
        if not name or not resume_path:
            messagebox.showerror("Error", "Please fill in all fields")
            return
            
        if not os.path.exists(resume_path):
            messagebox.showerror("Error", "Resume file not found")
            return
            
        try:
            # Process the resume and get scores
            scores = self.process_resume(resume_path)
            
            # Add to rankings data
            candidate_data = {
                'resume_file': os.path.basename(resume_path),
                'candidate_name': name,
                'transformer_score': scores['transformer_score'],
                'tfidf_score': scores['tfidf_score'],
                'section_score': scores['section_score'],
                'combined_score': scores['combined_score'],
                'section_details': scores['section_details']
            }
            
            self.rankings_data.append(candidate_data)
            
            # Sort rankings
            self.rankings_data.sort(key=lambda x: x['combined_score'], reverse=True)
            
            # Save updated data
            with open('hybrid_matching_results.json', 'w') as f:
                json.dump(self.rankings_data, f, indent=2)
                
            # Update display
            self.update_rankings_display()
            
            # Clear form
            self.name_var.set("")
            self.resume_path_var.set("")
            self.progress_var.set(0)
            
            messagebox.showinfo("Success", "Candidate added successfully!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to process resume: {str(e)}")
            
    def process_resume(self, resume_path):
        # Update progress
        self.progress_var.set(10)
        self.root.update()
        
        # Extract text from PDF
        doc = fitz.open(resume_path)
        text = ""
        for page in doc:
            text += page.get_text()
        
        self.progress_var.set(30)
        self.root.update()
        
        # Load job description
        try:
            with open('job_description.txt', 'r') as f:
                job_desc = f.read().strip()
        except FileNotFoundError:
            job_desc = "Sample job description"
        
        # Get transformer embeddings using sentence-transformers
        resume_embedding = self.model.encode(text, convert_to_tensor=True)
        job_embedding = self.model.encode(job_desc, convert_to_tensor=True)
        
        # Calculate transformer score using sentence-transformers util
        transformer_score = util.pytorch_cos_sim(job_embedding, resume_embedding)[0][0].item()
        
        self.progress_var.set(60)
        self.root.update()
        
        # Get TF-IDF score
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([text, job_desc])
        tfidf_score = sklearn_cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        
        self.progress_var.set(80)
        self.root.update()
        
        # Get section scores using spaCy
        doc = self.nlp(text)
        job_doc = self.nlp(job_desc)
        section_score = doc.similarity(job_doc)
        
        self.progress_var.set(100)
        self.root.update()
        
        return {
            'transformer_score': transformer_score,
            'tfidf_score': tfidf_score,
            'section_score': section_score,
            'combined_score': (
                transformer_score * 0.4 +
                tfidf_score * 0.3 +
                section_score * 0.3
            ),
            'section_details': {
                'experience': 0.8,
                'education': 0.7,
                'skills': 0.6
            }
        }
        
    def update_rankings_display(self):
        # Clear existing items
        for item in self.tree.get_children():
            self.tree.delete(item)
            
        # Add rankings data
        for i, candidate in enumerate(self.rankings_data, 1):
            self.tree.insert('', 'end', values=(
                i,
                candidate['candidate_name'],
                f"{candidate['combined_score']*100:.2f}%",
                f"{candidate['transformer_score']*100:.2f}%",
                f"{candidate['tfidf_score']*100:.2f}%",
                f"{candidate['section_score']*100:.2f}%"
            ))

if __name__ == "__main__":
    root = tk.Tk()
    app = ResumeRankingGUI(root)
    root.mainloop()