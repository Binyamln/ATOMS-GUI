import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext
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
from PIL import Image, ImageTk

class ModernResumeRankingGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("ATOMS - AI Powered Talent and Opportunity Matching System")
        
        # Get screen dimensions
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        
        # Set window size to 90% of screen size
        window_width = int(screen_width * 0.9)
        window_height = int(screen_height * 0.9)
        
        # Calculate position for center of screen
        position_x = (screen_width - window_width) // 2
        position_y = (screen_height - window_height) // 2
        
        # Set window size and position
        self.root.geometry(f"{window_width}x{window_height}+{position_x}+{position_y}")
        self.root.configure(bg='#1e1e1e')
        
        # Add maximize button functionality
        self.root.state('zoomed')  # This will maximize the window on Windows
        
        # Configure dark theme colors
        self.colors = {
            'bg': '#1e1e1e',
            'fg': '#ffffff',
            'accent': '#007acc',
            'secondary': '#252526',
            'highlight': '#2d2d2d',
            'text': '#cccccc'
        }
        
        # Configure styles for all widgets
        self.style = ttk.Style()
        self.style.theme_use('default')  # Start with default theme as base
        
        # Configure all common elements
        self.style.configure('.',  # This affects all widgets
                           background=self.colors['bg'],
                           foreground=self.colors['fg'],
                           fieldbackground=self.colors['secondary'],
                           troughcolor=self.colors['secondary'],
                           selectbackground=self.colors['accent'],
                           selectforeground=self.colors['fg'])
        
        # Configure specific widget styles
        self.style.configure('Dark.TFrame', background=self.colors['bg'])
        self.style.configure('Dark.TLabel', background=self.colors['bg'], foreground=self.colors['fg'])
        self.style.configure('Dark.TButton', background=self.colors['accent'], foreground=self.colors['fg'])
        self.style.configure('Dark.TEntry', fieldbackground=self.colors['secondary'], foreground=self.colors['fg'])
        
        # Configure Treeview colors (for rankings)
        self.style.configure('Treeview',
                           background=self.colors['secondary'],
                           foreground=self.colors['fg'],
                           fieldbackground=self.colors['secondary'])
        self.style.configure('Treeview.Heading',
                           background=self.colors['accent'],
                           foreground=self.colors['fg'])
        
        # Configure Text widget colors
        self.root.option_add('*Text*background', self.colors['secondary'])
        self.root.option_add('*Text*foreground', self.colors['fg'])
        
        # Configure Scrolledtext colors
        self.root.option_add('*ScrolledText*background', self.colors['secondary'])
        self.root.option_add('*ScrolledText*foreground', self.colors['fg'])
        
        # Load logo
        try:
            logo_img = Image.open("LOGO.png")
            logo_img = logo_img.resize((150, 150), Image.Resampling.LANCZOS)
            self.logo = ImageTk.PhotoImage(logo_img)
        except Exception as e:
            print(f"Error loading logo: {e}")
            self.logo = None
        
        # Create main container
        self.main_container = ttk.Frame(root, style='Dark.TFrame', padding="20")
        self.main_container.grid(row=0, column=0, sticky="nsew")
        
        # Configure grid weights
        root.grid_rowconfigure(0, weight=1)
        root.grid_columnconfigure(0, weight=1)
        
        # Create header with logo
        self.setup_header()
        
        # Initialize models
        self.model = SentenceTransformer('all-mpnet-base-v2')
        self.nlp = spacy.load("en_core_web_lg")
        
        # Load existing data
        self.load_data()
        
        # Create and setup tabs
        self.setup_tabs()
        
    def setup_header(self):
        header_frame = ttk.Frame(self.main_container, style='Dark.TFrame')
        header_frame.grid(row=0, column=0, sticky="ew", pady=(0, 20))
        
        # Logo on the left
        if self.logo:
            logo_label = ttk.Label(header_frame, image=self.logo, style='Dark.TLabel')
            logo_label.grid(row=0, column=0, rowspan=2, padx=(0, 20))
        
        # Title and description
        title_label = ttk.Label(
            header_frame,
            text="ATOMS Resume Ranking System",
            font=('Helvetica', 24, 'bold'),
            style='Dark.TLabel'
        )
        title_label.grid(row=0, column=1, sticky="w")
        
        desc_label = ttk.Label(
            header_frame,
            text="AI Powered Talent and Opportunity Matching System",
            font=('Helvetica', 12),
            style='Dark.TLabel'
        )
        desc_label.grid(row=1, column=1, sticky="w")
        
    def setup_tabs(self):
        # Configure notebook and tab styles with proper contrast
        self.style.configure('Dark.TNotebook',
                           background=self.colors['bg'])
        
        # Configure the tab style
        self.style.configure('Dark.TNotebook.Tab',
                           background=self.colors['secondary'],
                           foreground=self.colors['fg'],
                           padding=[10, 5])
        
        # Configure selected tab style
        self.style.map('Dark.TNotebook.Tab',
                      background=[('selected', self.colors['accent'])],
                      foreground=[('selected', self.colors['fg'])],
                      expand=[('selected', [1, 1, 1, 0])])
        
        # Create notebook with custom style
        self.tab_control = ttk.Notebook(self.main_container, style='Dark.TNotebook')
        self.tab_control.grid(row=1, column=0, sticky="nsew")
        
        # Create frames for each tab with dark background
        self.add_candidate_tab = ttk.Frame(self.tab_control, style='Dark.TFrame')
        self.rankings_tab = ttk.Frame(self.tab_control, style='Dark.TFrame')
        self.job_desc_tab = ttk.Frame(self.tab_control, style='Dark.TFrame')
        self.about_tab = ttk.Frame(self.tab_control, style='Dark.TFrame')
        
        # Add tabs to notebook
        self.tab_control.add(self.add_candidate_tab, text='Add Candidate')
        self.tab_control.add(self.rankings_tab, text='Rankings')
        self.tab_control.add(self.job_desc_tab, text='Job Description')
        self.tab_control.add(self.about_tab, text='About')
        
        # Setup individual tabs
        self.setup_add_candidate_tab()
        self.setup_rankings_tab()
        self.setup_job_desc_tab()
        self.setup_about_tab()
        
    def setup_about_tab(self):
        # Main frame for the about tab
        about_frame = ttk.Frame(self.about_tab, style='Dark.TFrame')
        about_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Create main scrollable frame
        main_canvas = tk.Canvas(
            about_frame,
            bg=self.colors['secondary'],
            highlightthickness=0
        )
        scrollbar = ttk.Scrollbar(about_frame, orient="vertical", command=main_canvas.yview)
        
        # Content frame that will hold all the information
        content_frame = ttk.Frame(main_canvas, style='Dark.TFrame')
        
        # Pack scrollbar and canvas
        scrollbar.pack(side="right", fill="y")
        main_canvas.pack(side="left", fill="both", expand=True)
        
        # Configure the canvas
        main_canvas.configure(yscrollcommand=scrollbar.set)
        main_canvas.create_window((0, 0), window=content_frame, anchor="nw", tags="content")
        
        # Project title
        title_label = ttk.Label(
            content_frame,
            text="ATOMS - AI Powered Talent and Opportunity Matching System",
            font=('Helvetica', 24, 'bold'),
            style='Dark.TLabel',
            wraplength=800
        )
        title_label.pack(fill="x", pady=(20, 40))
        
        # Team section
        team_frame = ttk.LabelFrame(
            content_frame,
            text="Project Team",
            style='Dark.TFrame',
            padding=20
        )
        team_frame.pack(fill="x", pady=(0, 40), padx=20)
        
        team_members = [
            "Nicolas Perez",
            "Mackenzi Mumford",
            "Edmundo Leets",
            "Christopher Clingerman",
            "John Meija",
            "Binyamin Abukar"
        ]
        
        for member in team_members:
            ttk.Label(
                team_frame,
                text=f"• {member}",
                style='Dark.TLabel',
                font=('Helvetica', 14)
            ).pack(anchor="w", pady=5)
        
        # Institution
        ttk.Label(
            content_frame,
            text="University of Louisville",
            font=('Helvetica', 18, 'italic'),
            style='Dark.TLabel'
        ).pack(pady=(0, 40))
        
        # Sponsors
        sponsor_frame = ttk.LabelFrame(
            content_frame,
            text="Sponsors",
            style='Dark.TFrame',
            padding=20
        )
        sponsor_frame.pack(fill="x", pady=(0, 40), padx=20)
        
        for sponsor in ["Betaflix", "John Garrison", "John Spurgeon"]:
            ttk.Label(
                sponsor_frame,
                text=sponsor,
                style='Dark.TLabel',
                font=('Helvetica', 14)
            ).pack(anchor="w", pady=5)
        
        # Project Description
        desc_frame = ttk.LabelFrame(
            content_frame,
            text="Project Description",
            style='Dark.TFrame',
            padding=20
        )
        desc_frame.pack(fill="x", pady=(0, 40), padx=20)
        
        description = """ATOMS is designed to revolutionize the hiring process by leveraging artificial intelligence to match candidates with job opportunities based on multi-dimensional analysis of skills, experience, and job descriptions. The system aims to mitigate inefficiencies in the hiring process by using artificial intelligence to accurately match candidates to opportunities that best utilize their skills."""
        
        desc_label = ttk.Label(
            desc_frame,
            text=description,
            style='Dark.TLabel',
            font=('Helvetica', 14),
            wraplength=800
        )
        desc_label.pack(fill="x", pady=10)
        
        # Technical Details
        tech_frame = ttk.LabelFrame(
            content_frame,
            text="Technical Implementation",
            style='Dark.TFrame',
            padding=20
        )
        tech_frame.pack(fill="x", pady=(0, 40), padx=20)
        
        tech_details = [
            "• Transformer-based neural network architecture",
            "• Python 3.12+ implementation",
            "• Docker-based containerized deployment",
            "• xAPI (JSON format) data storage",
            "• ML and NLP libraries for processing",
            "• Industry-standard security compliance (GDPR, CCPA)"
        ]
        
        for detail in tech_details:
            ttk.Label(
                tech_frame,
                text=detail,
                style='Dark.TLabel',
                font=('Helvetica', 14)
            ).pack(anchor="w", pady=5)
        
        # Configure scrolling
        def configure_scroll(event):
            # Update the scrollregion to encompass the entire content frame
            main_canvas.configure(scrollregion=main_canvas.bbox("all"))
            
            # Set the canvas width to match the frame width
            canvas_width = event.width
            main_canvas.itemconfig("content", width=canvas_width)
        
        content_frame.bind('<Configure>', configure_scroll)
        
        # Bind mouse wheel to scrolling
        def on_mousewheel(event):
            main_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        
        # Bind mouse wheel event to all relevant widgets
        main_canvas.bind_all("<MouseWheel>", on_mousewheel)
        
        # Bind frame destruction to remove mousewheel binding
        def on_destroy(event):
            main_canvas.unbind_all("<MouseWheel>")
        
        about_frame.bind("<Destroy>", on_destroy)

    def setup_other_tabs_sizing(self):
        # Adjust Rankings tab
        if hasattr(self, 'rankings_tab'):
            for child in self.rankings_tab.winfo_children():
                child.grid_configure(padx=40, pady=40)
        
        # Adjust Job Description tab
        if hasattr(self, 'job_desc_tab'):
            for child in self.job_desc_tab.winfo_children():
                child.grid_configure(padx=40, pady=40)
        
        # Adjust Add Candidate tab
        if hasattr(self, 'add_candidate_tab'):
            for child in self.add_candidate_tab.winfo_children():
                child.grid_configure(padx=40, pady=40)

    def load_data(self):
        try:
            with open('hybrid_matching_results.json', 'r') as f:
                self.rankings_data = json.load(f)
        except FileNotFoundError:
            self.rankings_data = []
            
    def setup_add_candidate_tab(self):
        # Create form frame with dark theme
        form_frame = ttk.LabelFrame(
            self.add_candidate_tab,
            text="Candidate Information",
            padding="20",
            style='Dark.TFrame'
        )
        form_frame.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
        
        # Name field
        ttk.Label(form_frame, text="Full Name:", style='Dark.TLabel').grid(
            row=0, column=0, sticky="w", pady=10
        )
        self.name_var = tk.StringVar()
        name_entry = ttk.Entry(
            form_frame,
            textvariable=self.name_var,
            width=40,
            style='Dark.TEntry'
        )
        name_entry.grid(row=0, column=1, sticky="w", pady=10)
        
        # Resume file selection
        ttk.Label(form_frame, text="Resume PDF:", style='Dark.TLabel').grid(
            row=1, column=0, sticky="w", pady=10
        )
        self.resume_path_var = tk.StringVar()
        resume_entry = ttk.Entry(
            form_frame,
            textvariable=self.resume_path_var,
            width=40,
            style='Dark.TEntry'
        )
        resume_entry.grid(row=1, column=1, sticky="w", pady=10)
        
        browse_btn = ttk.Button(
            form_frame,
            text="Browse",
            command=self.browse_resume,
            style='Dark.TButton'
        )
        browse_btn.grid(row=1, column=2, padx=10, pady=10)
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            form_frame,
            variable=self.progress_var,
            maximum=100,
            style='Dark.Horizontal.TProgressbar'
        )
        self.progress_bar.grid(row=2, column=0, columnspan=3, sticky="ew", pady=10)
        
        # Submit button
        submit_btn = ttk.Button(
            form_frame,
            text="Add Candidate",
            command=self.add_candidate,
            style='Dark.TButton'
        )
        submit_btn.grid(row=3, column=0, columnspan=3, pady=20)
        
    def setup_rankings_tab(self):
        rankings_frame = ttk.Frame(self.rankings_tab, style='Dark.TFrame', padding="20")
        rankings_frame.grid(row=0, column=0, sticky="nsew")
        
        # Main rankings table
        table_frame = ttk.LabelFrame(rankings_frame, text="All Rankings", style='Dark.TFrame', padding="10")
        table_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        
        columns = ('Rank', 'Name', 'File', 'Combined Score', 'Transformer Score', 'TFIDF Score', 'Section Score')
        self.tree = ttk.Treeview(table_frame, columns=columns, show='headings', style='Treeview')
        
        column_widths = {
            'Rank': 50,
            'Name': 300,
            'File': 150,
            'Combined Score': 120,
            'Transformer Score': 120,
            'TFIDF Score': 120,
            'Section Score': 120
        }
        
        for col in columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=column_widths[col], anchor='center')
        
        y_scrollbar = ttk.Scrollbar(table_frame, orient='vertical', command=self.tree.yview)
        x_scrollbar = ttk.Scrollbar(table_frame, orient='horizontal', command=self.tree.xview)
        self.tree.configure(yscrollcommand=y_scrollbar.set, xscrollcommand=x_scrollbar.set)
        
        self.tree.grid(row=0, column=0, sticky="nsew")
        y_scrollbar.grid(row=0, column=1, sticky="ns")
        x_scrollbar.grid(row=1, column=0, sticky="ew")
        
        # Recently Added section
        recent_frame = ttk.LabelFrame(rankings_frame, text="Recently Added", style='Dark.TFrame', padding="10")
        recent_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=10)
        
        self.recent_tree = ttk.Treeview(recent_frame, columns=columns, show='headings', style='Treeview')
        
        for col in columns:
            self.recent_tree.heading(col, text=col)
            self.recent_tree.column(col, width=column_widths[col], anchor='center')
        
        recent_y_scrollbar = ttk.Scrollbar(recent_frame, orient='vertical', command=self.recent_tree.yview)
        recent_x_scrollbar = ttk.Scrollbar(recent_frame, orient='horizontal', command=self.recent_tree.xview)
        self.recent_tree.configure(yscrollcommand=recent_y_scrollbar.set, xscrollcommand=recent_x_scrollbar.set)
        
        self.recent_tree.grid(row=0, column=0, sticky="nsew")
        recent_y_scrollbar.grid(row=0, column=1, sticky="ns")
        recent_x_scrollbar.grid(row=1, column=0, sticky="ew")
        
        rankings_frame.grid_rowconfigure(0, weight=3)
        rankings_frame.grid_rowconfigure(1, weight=1)
        rankings_frame.grid_columnconfigure(0, weight=1)
        
        ttk.Button(rankings_frame, text="Refresh Rankings", command=self.update_rankings_display).grid(row=2, column=0, pady=10)
        
        self.update_rankings_display()
        
    def setup_job_desc_tab(self):
        # Main frame for job descriptions
        job_desc_frame = ttk.Frame(self.job_desc_tab, style='Dark.TFrame')
        job_desc_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Create left panel for job list
        left_panel = ttk.Frame(job_desc_frame, style='Dark.TFrame')
        left_panel.pack(side="left", fill="y", padx=(0, 10))
        
        # Create job list label
        ttk.Label(
            left_panel,
            text="Available Jobs",
            font=('Helvetica', 16, 'bold'),
            style='Dark.TLabel'
        ).pack(pady=(0, 10))
        
        # Create listbox for jobs with scrollbar
        job_list_frame = ttk.Frame(left_panel, style='Dark.TFrame')
        job_list_frame.pack(fill="both", expand=True)
        
        self.job_listbox = tk.Listbox(
            job_list_frame,
            bg=self.colors['secondary'],
            fg=self.colors['fg'],
            font=('Helvetica', 12),
            selectmode="single",
            width=40,
            height=20
        )
        job_scrollbar = ttk.Scrollbar(job_list_frame, orient="vertical", command=self.job_listbox.yview)
        self.job_listbox.configure(yscrollcommand=job_scrollbar.set)
        
        self.job_listbox.pack(side="left", fill="both", expand=True)
        job_scrollbar.pack(side="right", fill="y")
        
        # Create right panel for job description
        right_panel = ttk.Frame(job_desc_frame, style='Dark.TFrame')
        right_panel.pack(side="right", fill="both", expand=True)
        
        # Create description area with label
        ttk.Label(
            right_panel,
            text="Job Description",
            font=('Helvetica', 16, 'bold'),
            style='Dark.TLabel'
        ).pack(pady=(0, 10))
        
        # Create text widget for job description
        self.job_desc_text = scrolledtext.ScrolledText(
            right_panel,
            wrap=tk.WORD,
            font=('Helvetica', 12),
            bg=self.colors['secondary'],
            fg=self.colors['fg'],
            height=30
        )
        self.job_desc_text.pack(fill="both", expand=True)
        
        # Add buttons for job description management
        button_frame = ttk.Frame(right_panel, style='Dark.TFrame')
        button_frame.pack(fill="x", pady=10)
        
        ttk.Button(
            button_frame,
            text="Save Changes",
            command=self.save_job_description,
            style='Dark.TButton'
        ).pack(side="right", padx=5)
        
        ttk.Button(
            button_frame,
            text="Add New Job",
            command=self.add_new_job,
            style='Dark.TButton'
        ).pack(side="right", padx=5)
        
        # Load existing jobs
        self.load_jobs()
        
        # Bind selection event
        self.job_listbox.bind('<<ListboxSelect>>', self.on_job_select)

    def load_jobs(self):
        try:
            with open("D:/ATOMS/jobfiles/normalized_jobs.json", 'r', encoding='utf-8') as file:
                self.jobs_data = json.load(file)
                
                # Clear existing items
                self.job_listbox.delete(0, tk.END)
                
                # Add jobs to listbox
                for job in self.jobs_data:
                    # Use file_name as the job title in the list
                    job_title = job.get('file_name', 'Untitled Job')
                    self.job_listbox.insert(tk.END, job_title)
                
                # Select first job if available
                if self.job_listbox.size() > 0:
                    self.job_listbox.selection_set(0)
                    self.on_job_select(None)
                    
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load jobs: {str(e)}")
            self.jobs_data = []

    def on_job_select(self, event):
        selection = self.job_listbox.curselection()
        if selection:
            index = selection[0]
            job_data = self.jobs_data[index]
            
            # Clear current text
            self.job_desc_text.delete('1.0', tk.END)
            
            # Format the text with sections
            formatted_text = f"Job Title: {job_data.get('file_name', '')}\n\n"
            
            sections = job_data.get('sections', {})
            
            # Add Contact/Overview section
            if 'contact' in sections:
                formatted_text += "Overview:\n"
                formatted_text += f"{sections['contact']}\n\n"
            
            # Add Experience Requirements
            if 'experience' in sections:
                formatted_text += "Experience Requirements:\n"
                formatted_text += f"{sections['experience']}\n\n"
            
            # Add Skills Requirements
            if 'skills' in sections:
                formatted_text += "Required Skills:\n"
                # Split skills by bullet points and format them
                skills = sections['skills'].split('•')
                for skill in skills:
                    if skill.strip():  # Only add non-empty skills
                        formatted_text += f"• {skill.strip()}\n"
                formatted_text += "\n"
            
            # Add Job Description/Interests
            if 'interests' in sections:
                formatted_text += "Job Description:\n"
                formatted_text += f"{sections['interests']}\n\n"
            
            self.job_desc_text.insert('1.0', formatted_text)
            self.job_desc_text.configure(state='normal')

    def save_job_description(self):
        selection = self.job_listbox.curselection()
        if selection:
            index = selection[0]
            
            # Get the text content
            content = self.job_desc_text.get('1.0', tk.END).strip()
            
            # Parse the content back into sections
            sections = {}
            current_section = None
            current_content = []
            
            for line in content.split('\n'):
                if line.endswith(':'):  # Section header
                    if current_section and current_content:
                        sections[current_section.lower()] = '\n'.join(current_content).strip()
                    current_section = line[:-1].lower()
                    current_content = []
                else:
                    current_content.append(line)
            
            # Add the last section
            if current_section and current_content:
                sections[current_section.lower()] = '\n'.join(current_content).strip()
            
            # Update the jobs data
            self.jobs_data[index]['sections'] = sections
            
            try:
                with open("D:/ATOMS/jobfiles/normalized_jobs.json", 'w', encoding='utf-8') as file:
                    json.dump(self.jobs_data, file, indent=2, ensure_ascii=False)
                messagebox.showinfo("Success", "Job description saved successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save job description: {str(e)}")

    def add_new_job(self):
        # Create a new window for job entry
        new_job_window = tk.Toplevel(self.root)
        new_job_window.title("Add New Job")
        new_job_window.configure(bg=self.colors['bg'])
        
        # Job title entry
        ttk.Label(
            new_job_window,
            text="Job Title:",
            style='Dark.TLabel'
        ).pack(pady=(10, 5))
        
        title_entry = ttk.Entry(new_job_window, width=50)
        title_entry.pack(padx=10, pady=(0, 10))
        
        # Job description entry
        ttk.Label(
            new_job_window,
            text="Job Description:",
            style='Dark.TLabel'
        ).pack(pady=(10, 5))
        
        desc_text = scrolledtext.ScrolledText(
            new_job_window,
            width=50,
            height=10,
            bg=self.colors['secondary'],
            fg=self.colors['fg']
        )
        desc_text.pack(padx=10, pady=(0, 10))
        
        def save_new_job():
            title = title_entry.get().strip()
            description = desc_text.get('1.0', tk.END).strip()
            
            if title and description:
                new_job = {
                    "title": title,
                    "description": description,
                    "requirements": [],
                    "qualifications": []
                }
                
                self.jobs_data.append(new_job)
                
                try:
                    with open("D:/ATOMS/jobfiles/normalized_jobs.json", 'w') as file:
                        json.dump(self.jobs_data, file, indent=2)
                    
                    self.job_listbox.insert(tk.END, title)
                    new_job_window.destroy()
                    messagebox.showinfo("Success", "New job added successfully!")
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to save new job: {str(e)}")
            else:
                messagebox.showerror("Error", "Please fill in both title and description!")
        
        ttk.Button(
            new_job_window,
            text="Save Job",
            command=save_new_job,
            style='Dark.TButton'
        ).pack(pady=10)

    def browse_resume(self):
        filename = filedialog.askopenfilename(
            title="Select Resume PDF",
            filetypes=(("PDF files", "*.pdf"), ("All files", "*.*"))
        )
        if filename:
            self.resume_path_var.set(filename)
            
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
        for item in self.tree.get_children():
            self.tree.delete(item)
        for item in self.recent_tree.get_children():
            self.recent_tree.delete(item)
        
        try:
            with open('hybrid_matching_results.json', 'r') as f:
                self.rankings_data = json.load(f)
            
            self.rankings_data.sort(key=lambda x: x['combined_score'], reverse=True)
            
            for i, candidate in enumerate(self.rankings_data, 1):
                combined_score = f"{candidate['combined_score']*100:.2f}%"
                transformer_score = f"{candidate['transformer_score']*100:.2f}%"
                tfidf_score = f"{candidate['tfidf_score']*100:.2f}%"
                section_score = f"{candidate['section_score']*100:.2f}%"
                
                values = (
                    i,
                    candidate['candidate_name'][:100],
                    candidate['resume_file'],
                    combined_score,
                    transformer_score,
                    tfidf_score,
                    section_score
                )
                
                self.tree.insert('', 'end', values=values)
                
                # Add to recent section if it's one of the last 5 added
                if i > len(self.rankings_data) - 5:
                    self.recent_tree.insert('', 0, values=values)
            
        except FileNotFoundError:
            messagebox.showwarning("Warning", "No rankings data found.")
        except Exception as e:
            messagebox.showerror("Error", f"Error loading rankings: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = ModernResumeRankingGUI(root)
    root.mainloop()