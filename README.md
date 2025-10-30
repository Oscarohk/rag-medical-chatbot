# **MediBot**

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
Open a browser and go to [localhost:8080](localhost:8080)

