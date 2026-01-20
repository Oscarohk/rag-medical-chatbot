# **MediBot**
<img width="1457" height="697" alt="螢幕擷取畫面 (250)" src="https://github.com/user-attachments/assets/a4689293-d190-4577-ac3b-16c919ed9d37" />

# Techstack
* Python
* LangChain
* Flask
* HuggingFace
* Pinecone

# How to run?

### STEP 01

Clone the repository

```bash
git clone https://github.com/Oscarohk/rag-medical-chatbot.git
```

### STEP 02 - Create a conda environment after opening the repository

```bash
conda create -n medibot python=3.10 -y
```

```bash
conda activate medibot
```

### STEP 03 - Install the requirements

```bash
pip install -r requirements.txt
```

### STEP 04 - Create a ```.env``` file in the root directory and add your Pinecone & HuggingFace credentials

```python
PINECONE_API_KEY = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
HUGGINGFACEHUB_API_TOKEN = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

### STEP 05 - Create the desired vector store using Pinecone

```bash
python store_index.py
```

### STEP 06 - Start the web app

```bash
python app.py
```

### FINAL STEP

Open a browser and go to _**localhost:8080**_
