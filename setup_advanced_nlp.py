"""
Setup script for Advanced Chinese NLP System
Installs dependencies and initializes models
"""

import subprocess
import sys
import os
import json
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_command(command, description=""):
    """Run a command and handle errors"""
    logger.info(f"Running: {description or command}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        if result.stdout:
            logger.info(f"Output: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running command: {e}")
        if e.stderr:
            logger.error(f"Error output: {e.stderr}")
        return False

def install_dependencies():
    """Install required Python packages"""
    logger.info("Installing Python dependencies...")
    
    # Basic requirements
    basic_requirements = [
        "flask>=2.3.0",
        "flask-cors>=4.0.0",
        "jieba>=0.42.1",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "requests>=2.31.0"
    ]
    
    # Advanced NLP requirements
    advanced_requirements = [
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "torchcrf>=1.1.0",
        "scikit-learn>=1.3.0",
        "datasets>=2.14.0",
        "accelerate>=0.20.0",
        "tqdm>=4.65.0",
        "tokenizers>=0.13.0"
    ]
    
    # Optional CKIP requirements (may need special handling)
    ckip_requirements = [
        "ckip-transformers>=0.3.2"
    ]
    
    # Install basic requirements
    for package in basic_requirements:
        if not run_command(f"pip install {package}", f"Installing {package}"):
            logger.warning(f"Failed to install {package}")
    
    # Install advanced requirements
    for package in advanced_requirements:
        if not run_command(f"pip install {package}", f"Installing {package}"):
            logger.warning(f"Failed to install {package}")
    
    # Try to install CKIP (may fail on some systems)
    logger.info("Attempting to install CKIP transformers...")
    for package in ckip_requirements:
        if not run_command(f"pip install {package}", f"Installing {package}"):
            logger.warning(f"CKIP installation failed - will use fallback methods")

def create_directories():
    """Create necessary directories"""
    logger.info("Creating directories...")
    
    directories = [
        "backend/nlp_models",
        "dict_cache",
        "model_cache",
        "chinese_ner_finetuned",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {directory}")

def download_pretrained_models():
    """Download pre-trained models if available"""
    logger.info("Checking for pre-trained models...")
    
    try:
        from transformers import AutoTokenizer, AutoModel
        
        # Download BERT base Chinese
        logger.info("Downloading BERT base Chinese...")
        tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
        model = AutoModel.from_pretrained("bert-base-chinese")
        logger.info("BERT base Chinese downloaded successfully")
        
        # Try to download RoBERTa Chinese (if available)
        try:
            logger.info("Attempting to download Chinese RoBERTa...")
            tokenizer_roberta = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
            model_roberta = AutoModel.from_pretrained("hfl/chinese-roberta-wwm-ext")
            logger.info("Chinese RoBERTa downloaded successfully")
        except Exception as e:
            logger.warning(f"Chinese RoBERTa not available: {e}")
        
    except ImportError:
        logger.warning("Transformers not available - skipping model download")
    except Exception as e:
        logger.error(f"Error downloading models: {e}")

def initialize_dictionary():
    """Initialize Chinese dictionary database"""
    logger.info("Initializing Chinese dictionary...")
    
    try:
        sys.path.append("backend")
        from nlp_models.chinese_dictionary import ChineseDictionary
        
        dictionary = ChineseDictionary(cache_dir="./dict_cache")
        stats = dictionary.get_dictionary_stats()
        
        logger.info(f"Dictionary initialized with {stats['words']} words and {stats['names']} names")
        
    except Exception as e:
        logger.error(f"Failed to initialize dictionary: {e}")

def run_basic_tests():
    """Run basic functionality tests"""
    logger.info("Running basic tests...")
    
    try:
        # Test basic NLP functionality
        test_text = "小明是一個學生，他每天和王老師一起學習。"
        
        # Test jieba segmentation
        import jieba
        seg_result = list(jieba.cut(test_text))
        logger.info(f"Jieba segmentation test: {' / '.join(seg_result)}")
        
        # Test dictionary if available
        try:
            sys.path.append("backend")
            from nlp_models.chinese_dictionary import ChineseDictionary
            
            dictionary = ChineseDictionary(cache_dir="./dict_cache")
            is_valid, confidence = dictionary.is_valid_name("小明")
            logger.info(f"Dictionary test - '小明' is valid: {is_valid}, confidence: {confidence:.2f}")
            
        except Exception as e:
            logger.warning(f"Dictionary test failed: {e}")
        
        # Test advanced NLP if available
        try:
            from backend.nlp_models.unified_nlp_pipeline import create_advanced_nlp_pipeline
            
            pipeline = create_advanced_nlp_pipeline(config_path="nlp_config.json")
            logger.info("Advanced NLP pipeline test successful")
            
        except Exception as e:
            logger.warning(f"Advanced NLP test failed: {e}")
        
        logger.info("Basic tests completed")
        
    except Exception as e:
        logger.error(f"Tests failed: {e}")

def create_startup_script():
    """Create a startup script for the enhanced application"""
    
    startup_script = '''#!/usr/bin/env python3
"""
Startup script for Advanced Chinese Character Analysis System
"""

import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from backend.app_enhanced import app
    print("Starting Advanced Chinese Character Analysis System...")
    app.run(host='0.0.0.0', port=5000, debug=False)
except ImportError:
    print("Advanced NLP not available, falling back to basic version...")
    from backend.app import app
    app.run(host='0.0.0.0', port=5000, debug=False)
'''
    
    with open("start_advanced.py", "w", encoding="utf-8") as f:
        f.write(startup_script)
    
    logger.info("Created startup script: start_advanced.py")

def main():
    """Main setup function"""
    logger.info("Starting Advanced Chinese NLP System Setup...")
    
    # Change to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # Setup steps
    steps = [
        ("Creating directories", create_directories),
        ("Installing dependencies", install_dependencies),
        ("Downloading pre-trained models", download_pretrained_models),
        ("Initializing dictionary", initialize_dictionary),
        ("Running basic tests", run_basic_tests),
        ("Creating startup script", create_startup_script)
    ]
    
    for step_name, step_func in steps:
        logger.info(f"Step: {step_name}")
        try:
            step_func()
            logger.info(f"✓ {step_name} completed")
        except Exception as e:
            logger.error(f"✗ {step_name} failed: {e}")
            logger.info("Continuing with next step...")
    
    logger.info("Setup completed!")
    logger.info("You can now start the system with: python start_advanced.py")
    logger.info("Or use the enhanced backend directly: python backend/app_enhanced.py")

if __name__ == "__main__":
    main()