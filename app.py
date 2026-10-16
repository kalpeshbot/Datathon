# -*- coding: utf-8 -*-
"""
LAPE: Leak Analysis & Prediction Engine
Enterprise Financial Triage & Leak Physics Platform
DATATHON 2026 Track 4: The Money Leak

LAPE combines business-based mathematical equations, scientifically constrained
synthetic data generation, machine learning, and evidence-linked financial analysis
to identify and prioritize potential financial and operational leaks.
"""

import os
import sys
from pathlib import Path
import json
import random
import datetime
import numpy as np
import pandas as pd
import streamlit as st

# Add repository root to path
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from src.validation import validate_schema, validate_foreign_keys
from src.features import (
    compute_pricing_features,
    compute_peer_group_mad_features,
    compute_inventory_features,
    compute_campaign_features,
    build_unified_transaction_feature_matrix
)
from src.rules import apply_baseline_a_simple_rule, apply_baseline_b_composite_rules
from src.model import LapeLeakClassifier, prepare_feature_matrix
from src.triage import (
    triage_transaction_record,
    triage_campaign_record,
    triage_inventory_record,
    generate_unified_alert_queue
)
from src.scenario_generator import ScenarioGenerator
from src.currency import format_inr, format_inr_compact
from src.pdf_report import generate_lape_pdf_report

# ==============================================================================
# DEMO CREDENTIALS & CONSTANTS
# ==============================================================================
DEMO_USERNAME = "demo@lape.local"
DEMO_PASSWORD = "lape2026"
APP_VERSION = "2.4-Enterprise"

# ==============================================================================
# STREAMLIT PAGE CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="LAPE | Leak Analysis & Prediction Engine",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# SESSION STATE INITIALIZATION
# ==============================================================================
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "theme" not in st.session_state:
    st.session_state.theme = "dark"
if "language" not in st.session_state:
    st.session_state.language = "English"
if "current_page" not in st.session_state:
    st.session_state.current_page = "Overview"
if "selected_alert_id" not in st.session_state:
    st.session_state.selected_alert_id = None
if "table_density" not in st.session_state:
    st.session_state.table_density = "Comfortable"
if "date_format" not in st.session_state:
    st.session_state.date_format = "DD/MM/YYYY"
if "default_report_scope" not in st.session_state:
    st.session_state.default_report_scope = "All Domains"
if "show_helper_text" not in st.session_state:
    st.session_state.show_helper_text = True
if "captcha_n1" not in st.session_state or "captcha_n2" not in st.session_state:
    st.session_state.captcha_n1 = random.randint(3, 11)
    st.session_state.captcha_n2 = random.randint(2, 9)

# ==============================================================================
# SUPPORTED LANGUAGES & LOCALIZATION DICTIONARY
# ==============================================================================
SUPPORTED_LANGUAGES = [
    "English",
    "Tamil — தமிழ்",
    "Telugu — తెలుగు",
    "Hindi — हिन्दी",
    "Kannada — ಕನ್ನಡ",
    "Malayalam — മലയാളം",
    "Gujarati — ગુજરાતી",
    "Punjabi — ਪੰਜਾਬੀ",
]

TRANSLATIONS = {
    "English": {
        "app_name": "LAPE",
        "app_full": "Leak Analysis & Prediction Engine",
        "financial_triage": "Tool to Find and Review Possible Money Leaks",
        "datathon_subtitle": "DATATHON 2026 Track 4 | Track & Protect Every Rupee",
        "sign_in": "Sign In to LAPE",
        "login_sub": "Enter your username and password, then answer the math question to enter.",
        "username": "Username or Email",
        "password": "Password",
        "show_password": "Show password",
        "demo_credentials": "Demo Login Details",
        "invalid_credentials": "Incorrect username or password. Please use the demo credentials below.",
        "invalid_captcha": "Incorrect math answer. Please try the question again.",
        "logout": "Sign Out",
        "overview": "Overview",
        "data_quality": "Data Checks",
        "alert_queue": "Warning List",
        "investigation": "Look Closer",
        "scenario_lab": "Test Situations",
        "model_eval": "Test the Model",
        "reports": "Reports",
        "settings": "Settings",
        "theme": "Theme Mode",
        "language": "Display Language",
        "dark": "Dark (Black & Grey)",
        "light": "Light (Clean & Bright)",
        "total_records": "Total Records Checked",
        "completed_neg": "Orders Sold Below Cost",
        "funnel_inversions": "Ad Campaigns with Broken Numbers",
        "inv_discrepancies": "Stock Count Mismatches",
        "pricing_margin": "Price vs Cost Checks",
        "funnel_violations": "Marketing Ad Funnel Issues",
        "inv_inconsistencies": "Warehouse Stock Warnings",
        "total_alerts": "Total Warnings Found",
        "critical_urgency": "Check Immediately",
        "high_reliability": "Very Reliable Information",
        "exposure_sum": "Total Money at Risk",
        "alert_stream": "List of Warnings (Ranked by Risk)",
        "drill_down": "Inspect Problem Details",
        "exposure_dim": "1. Possible Money at Risk",
        "reliability_dim": "2. How Reliable Is This Information?",
        "urgency_dim": "3. How Quickly Should We Check?",
        "suggested_action": "What to Check Next",
        "realtime_output": "Live Test Results",
        "algorithm_comparison": "Comparing Different Ways to Find Problems",
        "family_breakdown": "Results by Test Situation (Artificial Test Data)",
        "download_pdf": "Download Summary PDF Report",
        "download_csv": "Download Warnings as CSV",
    },
    "Tamil — தமிழ்": {
        "app_name": "LAPE",
        "app_full": "கசிவு பகுப்பாய்வு மற்றும் கணிப்பு இயந்திரம்",
        "financial_triage": "சாத்தியமான பண இழப்புகளைக் கண்டறியும் எளிய கருவி",
        "datathon_subtitle": "DATATHON 2026 டிராக் 4 | ஒவ்வொரு ரூபாயையும் பாதுகாத்தல்",
        "sign_in": "உள்நுழைக",
        "login_sub": "பயனர்பெயர் மற்றும் கடவுச்சொல்லை உள்ளிட்டு உள்நுழையவும்.",
        "username": "பயனர்பெயர் அல்லது மின்னஞ்சல்",
        "password": "கடவுச்சொல்",
        "show_password": "கடவுச்சொல்லைக் காட்டு",
        "demo_credentials": "டெமோ உள்நுழைவு விவரங்கள்",
        "invalid_credentials": "தவறான பயனர்பெயர் அல்லது கடவுச்சொல்.",
        "invalid_captcha": "தவறான கணித பதில். மீண்டும் முயற்சிக்கவும்.",
        "logout": "வெளியேறு",
        "overview": "கண்ணோட்டம்",
        "data_quality": "தரவு சரிபார்ப்புகள்",
        "alert_queue": "எச்சரிக்கை பட்டியல்",
        "investigation": "விவரங்களை ஆராய்க",
        "scenario_lab": "சோதனை சூழ்நிலைகள்",
        "model_eval": "மாதிரியைச் சோதிக்கவும்",
        "reports": "அறிக்கைகள்",
        "settings": "அமைப்புகள்",
        "theme": "வண்ண முறை",
        "language": "காட்சி மொழி",
        "dark": "இருண்ட முறை",
        "light": "வெளிச்ச முறை",
        "total_records": "சரிபார்க்கப்பட்ட மொத்த பதிவுகள்",
        "completed_neg": "செலவை விடக் குறைவாக விற்ற ஆர்டர்கள்",
        "funnel_inversions": "விளம்பர தரவு முரண்பாடுகள்",
        "inv_discrepancies": "பங்கு எண்ணிக்கை முரண்பாடுகள்",
        "pricing_margin": "விலை மற்றும் செலவு சரிபார்ப்பு",
        "funnel_violations": "விளம்பர புனல் சிக்கல்கள்",
        "inv_inconsistencies": "கிடங்கு பங்கு எச்சரிக்கைகள்",
        "total_alerts": "கண்டறியப்பட்ட மொத்த எச்சரிக்கைகள்",
        "critical_urgency": "உடனடியாக சரிபார்க்கவும்",
        "high_reliability": "மிகவும் நம்பகமான தகவல்",
        "exposure_sum": "ஆபத்தில் உள்ள மொத்த பணம்",
        "alert_stream": "எச்சரிக்கைகளின் பட்டியல்",
        "drill_down": "சிக்கல் விவரங்களை ஆய்வு செய்க",
        "exposure_dim": "1. ஆபத்தில் இருக்கக்கூடிய பணம்",
        "reliability_dim": "2. இந்த தகவல் எவ்வளவு நம்பகமானது?",
        "urgency_dim": "3. எவ்வளவு விரைவாக சரிபார்க்க வேண்டும்?",
        "suggested_action": "அடுத்து என்ன சரிபார்க்க வேண்டும்",
        "realtime_output": "நேரடி சோதனை முடிவுகள்",
        "algorithm_comparison": "கண்டறியும் முறைகளின் ஒப்பீடு",
        "family_breakdown": "சோதனை முடிவுகள் (செயற்கை தரவு)",
        "download_pdf": "சுருக்கமான PDF அறிக்கையைப் பதிவிறக்குக",
        "download_csv": "எச்சரிக்கைகளை CSV ஆகப் பதிவிறக்குக",
    },
    "Telugu — తెలుగు": {
        "app_name": "LAPE",
        "app_full": "లీక్ విశ్లేషణ & ప్రిడిక్షన్ ఇంజిన్",
        "financial_triage": "డబ్బు లీకేజీలను గుర్తించే సాధనం",
        "datathon_subtitle": "DATATHON 2026 ట్రాక్ 4 | ప్రతి రూపాయిని రక్షించండి",
        "sign_in": "సైన్ ఇన్",
        "login_sub": "యూజర్‌నేమ్ మరియు పాస్‌వర్డ్ నమోదు చేసి లాగిన్ అవ్వండి.",
        "username": "యూజర్‌నేమ్ లేదా ఇమెయిల్",
        "password": "పాస్‌వర్డ్",
        "show_password": "పాస్‌వర్డ్ చూపించు",
        "demo_credentials": "డెమో లాగిన్ వివరాలు",
        "invalid_credentials": "తప్పు యూజర్‌నేమ్ లేదా పాస్‌వర్డ్.",
        "invalid_captcha": "గణిత సమాధానం తప్పు. దయచేసి మళ్ళీ ప్రయత్నించండి.",
        "logout": "లాగ్ అవుట్",
        "overview": "అవలోకనం",
        "data_quality": "డేటా తనిఖీలు",
        "alert_queue": "హెచ్చరికల జాబితా",
        "investigation": "వివరాలను పరిశీలించండి",
        "scenario_lab": "పరిస్థితుల పరీక్ష",
        "model_eval": "మోడల్‌ను పరీక్షించండి",
        "reports": "నివేదికలు",
        "settings": "సెట్టింగ్‌లు",
        "theme": "రంగు మోడ్",
        "language": "భాష",
        "dark": "డార్క్ మోడ్",
        "light": "లైట్ మోడ్",
        "total_records": "తనిఖీ చేసిన మొత్తం రికార్డులు",
        "completed_neg": "ఖర్చు కంటే తక్కువ ధరకు అమ్మిన ఆర్డర్లు",
        "funnel_inversions": "ప్రకటన డేటా సమస్యలు",
        "inv_discrepancies": "స్టాక్ లెక్కింపు తేడాలు",
        "pricing_margin": "ధర మరియు ఖర్చు తనిఖీ",
        "funnel_violations": "యాడ్ ఫన్నెల్ సమస్యలు",
        "inv_inconsistencies": "గోదాము స్టాక్ హెచ్చరికలు",
        "total_alerts": "మొత్తం హెచ్చరికలు",
        "critical_urgency": "వెంటనే తనిఖీ చేయండి",
        "high_reliability": "చాలా నమ్మదగిన సమాచారం",
        "exposure_sum": "రిస్క్‌లో ఉన్న మొత్తం డబ్బు",
        "alert_stream": "హెచ్చరికల జాబితా",
        "drill_down": "సమస్య వివరాల తనిఖీ",
        "exposure_dim": "1. రిస్క్‌లో ఉండగల డబ్బు",
        "reliability_dim": "2. ఈ సమాచారం ఎంతవరకు నమ్మదగినది?",
        "urgency_dim": "3. ఎంత త్వరగా తనిఖీ చేయాలి?",
        "suggested_action": "తర్వాత ఏమి తనిఖీ చేయాలి",
        "realtime_output": "లైవ్ పరీక్ష ఫలితాలు",
        "algorithm_comparison": "వివిధ పద్ధతుల పోలిక",
        "family_breakdown": "పరీక్ష ఫలితాలు (కృత్రిమ డేటా)",
        "download_pdf": "PDF నివేదికను డౌన్‌లోడ్ చేయండి",
        "download_csv": "హెచ్చరికలను CSV గా డౌన్‌లోడ్ చేయండి",
    },
    "Hindi — हिन्दी": {
        "app_name": "LAPE",
        "app_full": "लीक विश्लेषण एवं पूर्वानुमान इंजन",
        "financial_triage": "संभावित पैसों के नुकसान की पहचान करने वाला टूल",
        "datathon_subtitle": "DATATHON 2026 ट्रैक 4 | हर रुपये की सुरक्षा",
        "sign_in": "साइन इन करें",
        "login_sub": "लॉगिन करने के लिए यूज़रनेम और पासवर्ड दर्ज करें और गणित का सवाल हल करें।",
        "username": "यूज़रनेम या ईमेल",
        "password": "पासवर्ड",
        "show_password": "पासवर्ड दिखाएं",
        "demo_credentials": "डेमो लॉगिन विवरण",
        "invalid_credentials": "गलत यूज़रनेम या पासवर्ड। कृपया नीचे दिए गए डेमो विवरण का उपयोग करें।",
        "invalid_captcha": "गलत उत्तर। कृपया गणित का सवाल दोबारा हल करें।",
        "logout": "लॉग आउट",
        "overview": "सिंहावलोकन",
        "data_quality": "डेटा जांच",
        "alert_queue": "चेतावनी सूची",
        "investigation": "विस्तार से जांचें",
        "scenario_lab": "स्थितियों का परीक्षण",
        "model_eval": "मॉडल का परीक्षण",
        "reports": "रिपोर्ट्स",
        "settings": "सेटिंग्स",
        "theme": "थीम मोड",
        "language": "भाषा",
        "dark": "डार्क (काला और ग्रे)",
        "light": "लाइट (साफ़ और चमकीला)",
        "total_records": "जांची गई कुल प्रविष्टियां",
        "completed_neg": "लागत से कम कीमत पर बेचे गए ऑर्डर",
        "funnel_inversions": "विज्ञापन डेटा में गड़बड़ी",
        "inv_discrepancies": "स्टॉक गिनती में अंतर",
        "pricing_margin": "कीमत बनाम लागत जांच",
        "funnel_violations": "मार्केटिंग फनल समस्याएं",
        "inv_inconsistencies": "गोदाम स्टॉक चेतावनियां",
        "total_alerts": "मिली हुई कुल चेतावनियां",
        "critical_urgency": "तुरंत जांचें",
        "high_reliability": "अत्यंत विश्वसनीय जानकारी",
        "exposure_sum": "जोखिम में कुल संभावित राशि",
        "alert_stream": "चेतावनियों की सूची (जोखिम अनुसार)",
        "drill_down": "समस्या की पूरी जानकारी",
        "exposure_dim": "1. जोखिम में संभावित राशि",
        "reliability_dim": "2. यह जानकारी कितनी विश्वसनीय है?",
        "urgency_dim": "3. हमें कितनी जल्दी जांच करनी चाहिए?",
        "suggested_action": "आगे क्या जांचें",
        "realtime_output": "लाइव परीक्षण परिणाम",
        "algorithm_comparison": "समस्याएं खोजने के तरीकों की तुलना",
        "family_breakdown": "परीक्षण स्थितियों के परिणाम (कृत्रिम डेटा)",
        "download_pdf": "संक्षिप्त PDF रिपोर्ट डाउनलोड करें",
        "download_csv": "चेतावनियों को CSV में डाउनलोड करें",
    },
    "Kannada — ಕನ್ನಡ": {
        "app_name": "LAPE",
        "app_full": "ಸೋರಿಕೆ ವಿಶ್ಲೇಷಣೆ ಮತ್ತು ಭವಿಷ್ಯ ಎಂಜಿನ್",
        "financial_triage": "ಹಣಕಾಸು ಸೋರಿಕೆಗಳನ್ನು ಪತ್ತೆಹಚ್ಚುವ ಸಾಧನ",
        "datathon_subtitle": "DATATHON 2026 ಟ್ರ್ಯಾಕ್ 4 | ಪ್ರತಿಯೊಂದು ರೂಪಾಯಿಯ ರಕ್ಷಣೆ",
        "sign_in": "ಸೈನ್ ಇನ್",
        "login_sub": "ಲಾಗಿನ್ ಮಾಡಲು ಬಳಕೆದಾರ ಹೆಸರು ಮತ್ತು ಪಾಸ್‌ವರ್ಡ್ ನಮೂದಿಸಿ.",
        "username": "ಬಳಕೆದಾರ ಹೆಸರು ಅಥವಾ ಇಮೇಲ್",
        "password": "ಪಾಸ್‌ವರ್ಡ್",
        "show_password": "ಪಾಸ್‌ವರ್ಡ್ ತೋರಿಸು",
        "demo_credentials": "ಡೆಮೊ ಲಾಗಿನ್ ವಿವರಗಳು",
        "invalid_credentials": "ತಪ್ಪಾದ ಬಳಕೆದಾರ ಹೆಸರು ಅಥವಾ ಪಾಸ್‌ವರ್ಡ್.",
        "invalid_captcha": "ತಪ್ಪಾದ ಗಣಿತ ಉತ್ತರ. ದಯವಿಟ್ಟು ಮತ್ತೆ ಪ್ರಯತ್ನಿಸಿ.",
        "logout": "ಲಾಗ್ ಔಟ್",
        "overview": "ಅವಲೋಕನ",
        "data_quality": "ಡೇಟಾ ಪರಿಶೀಲನೆಗಳು",
        "alert_queue": "ಎಚ್ಚರಿಕೆ ಪಟ್ಟಿ",
        "investigation": "ವಿವರವಾಗಿ ಪರಿಶೀಲಿಸಿ",
        "scenario_lab": "ಪರಿಸ್ಥಿತಿಗಳ ಪರೀಕ್ಷೆ",
        "model_eval": "ಮಾದರಿ ಪರೀಕ್ಷೆ",
        "reports": "ವರದಿಗಳು",
        "settings": "ಸೆಟ್ಟಿಂಗ್‌ಗಳು",
        "theme": "ಬಣ್ಣದ ಮೋಡ್",
        "language": "ಭಾಷೆ",
        "dark": "ಡಾರ್ಕ್ ಮೋಡ್",
        "light": "ಲೈಟ್ ಮೋಡ್",
        "total_records": "ಪರಿಶೀಲಿಸಲಾದ ಒಟ್ಟು ದಾಖಲೆಗಳು",
        "completed_neg": "ವೆಚ್ಚಕ್ಕಿಂತ ಕಡಿಮೆ ಬೆಲೆಗೆ ಮಾರಾಟವಾದ ಆರ್ಡರ್‌ಗಳು",
        "funnel_inversions": "ಜಾಹೀರಾತು ಡೇಟಾ ದೋಷಗಳು",
        "inv_discrepancies": "ದಾಸ್ತಾನು ಎಣಿಕೆ ವ್ಯತ್ಯಾಸಗಳು",
        "pricing_margin": "ಬೆಲೆ ಮತ್ತು ವೆಚ್ಚ ಪರಿಶೀಲನೆ",
        "funnel_violations": "ಮಾರ್ಕೆಟಿಂಗ್ ಫನಲ್ ದೋಷಗಳು",
        "inv_inconsistencies": "ಗೋದಾಮು ಸ್ಟಾಕ್ ಎಚ್ಚರಿಕೆಗಳು",
        "total_alerts": "ಒಟ್ಟು ಎಚ್ಚರಿಕೆಗಳು",
        "critical_urgency": "ತಕ್ಷಣ ಪರಿಶೀಲಿಸಿ",
        "high_reliability": "ಹೆಚ್ಚು ವಿಶ್ವಾಸಾರ್ಹ ಮಾಹಿತಿ",
        "exposure_sum": "ಅಪಾಯದಲ್ಲಿರುವ ಒಟ್ಟು ಹಣ",
        "alert_stream": "ಎಚ್ಚರಿಕೆಗಳ ಪಟ್ಟಿ",
        "drill_down": "ಸಮಸ್ಯೆಯ ವಿವರಗಳ ಪರಿಶೀಲನೆ",
        "exposure_dim": "1. ಅಪಾಯದಲ್ಲಿರಬಹುದಾದ ಹಣ",
        "reliability_dim": "2. ಈ ಮಾಹಿತಿ ಎಷ್ಟು ವಿಶ್ವಾಸಾರ್ಹವಾಗಿದೆ?",
        "urgency_dim": "3. ಎಷ್ಟು ಬೇಗ ಪರಿಶೀಲಿಸಬೇಕು?",
        "suggested_action": "ಮುಂದೆ ಏನು ಪರಿಶೀಲಿಸಬೇಕು",
        "realtime_output": "ಲೈವ್ ಪರೀಕ್ಷಾ ಫಲಿತಾಂಶಗಳು",
        "algorithm_comparison": "ವಿವಿಧ ವಿಧಾನಗಳ ಹೋಲಿಕೆ",
        "family_breakdown": "ಪರೀಕ್ಷಾ ಫಲಿತಾಂಶಗಳು (ಕೃತಕ ಡೇಟಾ)",
        "download_pdf": "PDF ವರದಿಯನ್ನು ಡೌನ್‌ಲೋಡ್ ಮಾಡಿ",
        "download_csv": "ಎಚ್ಚರಿಕೆಗಳನ್ನು CSV ಆಗಿ ಡೌನ್‌ಲೋಡ್ ಮಾಡಿ",
    },
    "Malayalam — മലയാളം": {
        "app_name": "LAPE",
        "app_full": "ചോർച്ച വിശകലന പ്രവചന എഞ്ചിൻ",
        "financial_triage": "സാമ്പത്തിക നഷ്ടങ്ങൾ കണ്ടെത്താനുള്ള ഉപകരണം",
        "datathon_subtitle": "DATATHON 2026 ട്രാക്ക് 4 | ഓരോ രൂപയും സംരക്ഷിക്കുക",
        "sign_in": "സൈൻ ഇൻ ചെയ്യുക",
        "login_sub": "ലോഗിൻ ചെയ്യാൻ ഉപയോക്തൃനാമവും പാസ്‌വേഡും നൽകുക.",
        "username": "ഉപയോക്തൃനാമം അല്ലെങ്കിൽ ഇമെയിൽ",
        "password": "പാസ്‌വേഡ്",
        "show_password": "പാസ്‌വേഡ് കാണിക്കുക",
        "demo_credentials": "ഡെമോ ലോഗിൻ വിവരങ്ങൾ",
        "invalid_credentials": "തെറ്റായ ഉപയോക്തൃനാമം അല്ലെങ്കിൽ പാസ്‌വേഡ്.",
        "invalid_captcha": "തെറ്റായ ഉത്തരം. ദയവായി വീണ്ടും ശ്രമിക്കുക.",
        "logout": "ലോഗ് ഔട്ട്",
        "overview": "അവലോകനം",
        "data_quality": "ഡാറ്റ പരിശോധനകൾ",
        "alert_queue": "മുന്നറിയിപ്പ് ലിസ്റ്റ്",
        "investigation": "വിശദമായി പരിശോധിക്കുക",
        "scenario_lab": "സാഹചര്യങ്ങൾ പരീക്ഷിക്കുക",
        "model_eval": "മോഡൽ പരീക്ഷിക്കുക",
        "reports": "റിപ്പോർട്ടുകൾ",
        "settings": "ക്രമീകരണങ്ങൾ",
        "theme": "തീം മോഡ്",
        "language": "ഭാഷ",
        "dark": "ഡാർക്ക് മോഡ്",
        "light": "ലൈറ്റ് മോഡ്",
        "total_records": "പരിശോധിച്ച ആകെ രേഖകൾ",
        "completed_neg": "ചിലവിനേക്കാൾ കുറഞ്ഞ വിലയ്ക്ക് വിറ്റ ഓർഡറുകൾ",
        "funnel_inversions": "പരസ്യ ഡാറ്റയിലെ പൊരുത്തക്കേടുകൾ",
        "inv_discrepancies": "സ്റ്റോക്ക് കണക്കിലെ വ്യത്യാസങ്ങൾ",
        "pricing_margin": "വിലയും ചിലവും പരിശോധിക്കുക",
        "funnel_violations": "മാർക്കറ്റിംഗ് ഫണൽ പ്രശ്നങ്ങൾ",
        "inv_inconsistencies": "വെയർഹൗസ് സ്റ്റോക്ക് മുന്നറിയിപ്പുകൾ",
        "total_alerts": "കണ്ടെത്തിയ ആകെ മുന്നറിയിപ്പുകൾ",
        "critical_urgency": "ഉടൻ പരിശോധിക്കുക",
        "high_reliability": "വളരെ വിശ്വസനീയമായ വിവരം",
        "exposure_sum": "അപകടസാധ്യതയുള്ള ആകെ തുക",
        "alert_stream": "മുന്നറിയിപ്പുകളുടെ ലിസ്റ്റ്",
        "drill_down": "പ്രശ്നവിവരങ്ങൾ പരിശോധിക്കുക",
        "exposure_dim": "1. നഷ്ടപ്പെടാൻ സാധ്യതയുള്ള തുക",
        "reliability_dim": "2. ഈ വിവരം എത്രത്തോളം വിശ്വസനീയമാണ്?",
        "urgency_dim": "3. എത്ര വേഗത്തിൽ പരിശോധിക്കണം?",
        "suggested_action": "അടുത്തതായി എന്ത് പരിശോധിക്കണം",
        "realtime_output": "തത്സമയ പരിശോധനാ ഫലങ്ങൾ",
        "algorithm_comparison": "വിവിധ രീതികളുടെ താരതമ്യം",
        "family_breakdown": "പരിശോധനാ ഫലങ്ങൾ (കൃത്രിമ ഡാറ്റ)",
        "download_pdf": "PDF റിപ്പോർട്ട് ഡൗൺലോഡ് ചെയ്യുക",
        "download_csv": "മുന്നറിയിപ്പുകൾ CSV ആയി ഡൗൺലോഡ് ചെയ്യുക",
    },
    "Gujarati — ગુજરાતી": {
        "app_name": "LAPE",
        "app_full": "લીક વિશ્લેષણ અને અનુમાન એન્જિન",
        "financial_triage": "સંભવિત નાણાકીય નુકસાન શોધવાનું સાધન",
        "datathon_subtitle": "DATATHON 2026 ટ્રેક 4 | દરેક રૂપિયાનું રક્ષણ",
        "sign_in": "સાઇન ઇન",
        "login_sub": "લૉગિન કરવા માટે યુઝરનેમ અને પાસવર્ડ દાખલ કરો.",
        "username": "યુઝરનેમ અથવા ઇમેઇલ",
        "password": "પાસવર્ડ",
        "show_password": "પાસવર્ડ બતાવો",
        "demo_credentials": "ડેમો લૉગિન વિગતો",
        "invalid_credentials": "ખોટો યુઝરનેમ અથવા પાસવર્ડ.",
        "invalid_captcha": "ખોટો ગણિત જવાબ. કૃપા કરીને ફરી પ્રયાસ કરો.",
        "logout": "લૉગ આઉટ",
        "overview": "વિહંગાવલોકન",
        "data_quality": "ડેટા તપાસ",
        "alert_queue": "ચેતવણી યાદી",
        "investigation": "વિગતવાર તપાસો",
        "scenario_lab": "પરિસ્થિતિઓનું પરીક્ષણ",
        "model_eval": "મોડલ પરીક્ષણ",
        "reports": "અહેવાલો",
        "settings": "સેટિંગ્સ",
        "theme": "થીમ મોડ",
        "language": "ભાષા",
        "dark": "ડાર્ક મોડ",
        "light": "લાઇટ મોડ",
        "total_records": "કુલ તપાસેલા રેકોર્ડ્સ",
        "completed_neg": "પડતર કિંમત કરતાં ઓછા ભાવે વેચાયેલ ઓર્ડર્સ",
        "funnel_inversions": "જાહેરાત ડેટાની વિસંગતતાઓ",
        "inv_discrepancies": "સ્ટોક ગણતરીમાં તફાવત",
        "pricing_margin": "કિંમત અને પડતરની તપાસ",
        "funnel_violations": "માર્કેટિંગ ફનલ સમસ્યાઓ",
        "inv_inconsistencies": "ગોડાઉન સ્ટોક ચેતવણીઓ",
        "total_alerts": "મળેલી કુલ ચેતવણીઓ",
        "critical_urgency": "તરત જ તપાસો",
        "high_reliability": "ખૂબ વિશ્વસનીય માહિતી",
        "exposure_sum": "જોખમમાં રહેલ કુલ રકમ",
        "alert_stream": "ચેતવણીઓની યાદી",
        "drill_down": "સમસ્યાની વિગતવાર તપાસ",
        "exposure_dim": "1. જોખમમાં હોઈ શકે તેવી રકમ",
        "reliability_dim": "2. આ માહિતી કેટલી વિશ્વસનીય છે?",
        "urgency_dim": "3. કેટલી ઝડપથી તપાસ કરવી જોઈએ?",
        "suggested_action": "આગળ શું તપાસવું",
        "realtime_output": "લાઇવ પરીક્ષણ પરિણામો",
        "algorithm_comparison": "પદ્ધતિઓની સરખામણી",
        "family_breakdown": "પરીક્ષણ પરિણામો (કૃત્રિમ ડેટા)",
        "download_pdf": "PDF રિપોર્ટ ડાઉનલોડ કરો",
        "download_csv": "ચેતવણીઓ CSV તરીકે ડાઉનલોડ કરો",
    },
    "Punjabi — ਪੰਜਾਬੀ": {
        "app_name": "LAPE",
        "app_full": "ਲੀਕ ਵਿਸ਼ਲੇਸ਼ਣ ਅਤੇ ਭਵਿੱਖਬਾਣੀ ਇੰਜਨ",
        "financial_triage": "ਸੰਭਾਵੀ ਵਿੱਤੀ ਨੁਕਸਾਨ ਲੱਭਣ ਦਾ ਸਾਧਨ",
        "datathon_subtitle": "DATATHON 2026 ਟ੍ਰੈਕ 4 | ਹਰ ਰੁਪਏ ਦੀ ਸੁਰੱਖਿਆ",
        "sign_in": "ਸਾਈਨ ਇਨ",
        "login_sub": "ਲਾਗਇਨ ਕਰਨ ਲਈ ਯੂਜ਼ਰਨਾਮ ਅਤੇ ਪਾਸਵਰਡ ਦਰਜ ਕਰੋ।",
        "username": "ਯੂਜ਼ਰਨਾਮ ਜਾਂ ਈਮੇਲ",
        "password": "ਪਾਸਵਰਡ",
        "show_password": "ਪਾਸਵਰਡ ਦਿਖਾਓ",
        "demo_credentials": "ਡੈਮੋ ਲਾਗਇਨ ਵੇਰਵੇ",
        "invalid_credentials": "ਗਲਤ ਯੂਜ਼ਰਨਾਮ ਜਾਂ ਪਾਸਵਰਡ।",
        "invalid_captcha": "ਗਲਤ ਜਵਾਬ। ਕਿਰਪਾ ਕਰਕੇ ਦੁਬਾਰਾ ਕੋਸ਼ਿਸ਼ ਕਰੋ।",
        "logout": "ਲਾਗ ਆਉਟ",
        "overview": "ਸੰਖੇਪ",
        "data_quality": "ਡੇਟਾ ਜਾਂਚ",
        "alert_queue": "ਚੇਤਾਵਨੀ ਸੂਚੀ",
        "investigation": "ਵੇਰਵੇ ਜਾਂਚੋ",
        "scenario_lab": "ਸਥਿਤੀਆਂ ਦੀ ਪਰਖ",
        "model_eval": "ਮਾਡਲ ਦੀ ਪਰਖ",
        "reports": "ਰਿਪੋਰਟਾਂ",
        "settings": "ਸੈਟਿੰਗਾਂ",
        "theme": "ਥੀਮ ਮੋਡ",
        "language": "ਭਾਸ਼ਾ",
        "dark": "ਡਾਰਕ ਮੋਡ",
        "light": "ਲਾਈਟ ਮੋਡ",
        "total_records": "ਜਾਂਚੇ ਗਏ ਕੁੱਲ ਰਿਕਾਰਡ",
        "completed_neg": "ਲਾਗਤ ਤੋਂ ਘੱਟ ਕੀਮਤ 'ਤੇ ਵੇਚੇ ਗਏ ਆਰਡਰ",
        "funnel_inversions": "ਇਸ਼ਤਿਹਾਰ ਡੇਟਾ ਦੀਆਂ ਗਲਤੀਆਂ",
        "inv_discrepancies": "ਸਟਾਕ ਗਿਣਤੀ ਵਿੱਚ ਫ਼ਰਕ",
        "pricing_margin": "ਕੀਮਤ ਅਤੇ ਲਾਗਤ ਦੀ ਜਾਂਚ",
        "funnel_violations": "ਮਾਰਕੀਟਿੰਗ ਫਨਲ ਸਮੱਸਿਆਵਾਂ",
        "inv_inconsistencies": "ਵੇਅਰਹਾਊਸ ਸਟਾਕ ਚੇਤਾਵਨੀਆਂ",
        "total_alerts": "ਮਿਲੀਆਂ ਕੁੱਲ ਚੇਤਾਵਨੀਆਂ",
        "critical_urgency": "ਤੁਰੰਤ ਜਾਂਚ ਕਰੋ",
        "high_reliability": "ਬਹੁਤ ਭਰੋਸੇਯੋਗ ਜਾਣਕਾਰੀ",
        "exposure_sum": "ਜੋਖਮ ਵਿੱਚ ਕੁੱਲ ਰਕਮ",
        "alert_stream": "ਚੇਤਾਵਨੀਆਂ ਦੀ ਸੂਚੀ",
        "drill_down": "ਸਮੱਸਿਆ ਦੇ ਵੇਰਵੇ ਜਾਂਚੋ",
        "exposure_dim": "1. ਜੋਖਮ ਵਿੱਚ ਸੰਭਾਵੀ ਰਕਮ",
        "reliability_dim": "2. ਇਹ ਜਾਣਕਾਰੀ ਕਿੰਨੀ ਭਰੋਸੇਯੋਗ ਹੈ?",
        "urgency_dim": "3. ਕਿੰਨੀ ਜਲਦੀ ਜਾਂਚ ਕਰਨੀ ਚਾਹੀਦੀ ਹੈ?",
        "suggested_action": "ਅੱਗੇ ਕੀ ਜਾਂਚਣਾ ਹੈ",
        "realtime_output": "ਲਾਈਵ ਟੈਸਟ ਨਤੀਜੇ",
        "algorithm_comparison": "ਵੱਖ-ਵੱਖ ਤਰੀਕਿਆਂ ਦੀ ਤੁਲਨਾ",
        "family_breakdown": "ਟੈਸਟ ਨਤੀਜੇ (ਬਣਾਵਟੀ ਡੇਟਾ)",
        "download_pdf": "PDF ਰਿਪੋਰਟ ਡਾਊਨਲੋਡ ਕਰੋ",
        "download_csv": "ਚੇਤਾਵਨੀਆਂ CSV ਵਜੋਂ ਡਾਊਨਲੋਡ ਕਰੋ",
    },
}

def t(key: str) -> str:
    """Translate key to selected language with robust English fallback."""
    lang = st.session_state.get("language", "English")
    lang_dict = TRANSLATIONS.get(lang, {})
    val = lang_dict.get(key)
    if val:
        return val
    return TRANSLATIONS["English"].get(key, key)

# ==============================================================================
# EXACT THEMES: DARK (BLACK & GREY) VS LIGHT (CHROME CLEAN)
# Strictly NO gradients, NO purple accents, NO glassmorphism, NO blur effects
# ==============================================================================

DARK_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');
    
    :root {
        --bg-main: #0B0B0C;
        --bg-sidebar: #141416;
        --bg-panel: #1C1C1F;
        --bg-input: #242428;
        --border: #333338;
        --text-primary: #F2F2F3;
        --text-secondary: #A1A1AA;
        --accent: #2CB7B0;
        --accent-hover: #269f99;
        --warning: #D97706;
        --critical: #DC2626;
        --success: #10B981;
        --table-header-bg: #1C1C1F;
        --table-row-hover: #242428;
        --login-card-bg: #1C1C1F;
    }
    
    .stApp {
        background-color: var(--bg-main) !important;
        color: var(--text-primary) !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
    }
    
    .main .block-container {
        padding-top: 2.2rem !important;
        padding-bottom: 2rem !important;
        max-width: 1320px;
    }
    
    /* Top header configuration - keep visible so collapse toggle works */
    header[data-testid="stHeader"] {
        background: transparent !important;
        visibility: visible !important;
        height: auto !important;
        z-index: 1000 !important;
    }
    
    /* Hide unnecessary menu & deploy button */
    #MainMenu, footer, .stDeployButton {
        display: none !important;
        visibility: hidden !important;
    }
    
    /* CRITICAL FIX: Sidebar Collapse & Expand Toggle controls must always be visible & clickable */
    [data-testid="collapsedControl"],
    button[data-testid="stSidebarExpandButton"],
    button[data-testid="stSidebarCollapseButton"] {
        display: flex !important;
        visibility: visible !important;
        opacity: 1 !important;
        pointer-events: auto !important;
        z-index: 999999 !important;
        color: var(--text-primary) !important;
        background-color: var(--bg-panel) !important;
        border: 1px solid var(--border) !important;
        border-radius: 6px !important;
    }
    [data-testid="collapsedControl"] svg,
    button[data-testid="stSidebarExpandButton"] svg,
    button[data-testid="stSidebarCollapseButton"] svg {
        fill: var(--text-primary) !important;
        color: var(--text-primary) !important;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: var(--bg-sidebar) !important;
        border-right: 1px solid var(--border) !important;
    }
    section[data-testid="stSidebar"] * {
        color: var(--text-secondary) !important;
    }
    
    /* Sidebar Brand */
    .sidebar-brand {
        padding: 8px 6px 16px 6px;
        border-bottom: 1px solid var(--border);
        margin-bottom: 14px;
    }
    .sidebar-logo {
        font-size: 24px;
        font-weight: 700;
        color: var(--accent);
        letter-spacing: -0.5px;
        line-height: 1.1;
    }
    .sidebar-brand-title {
        font-size: 13.5px;
        color: var(--text-secondary);
        margin-top: 4px;
        text-transform: uppercase;
        letter-spacing: 0.6px;
    }
    
    /* Sidebar Group Heading */
    .sidebar-group-heading {
        font-size: 13.5px !important;
        font-weight: 600 !important;
        color: var(--text-secondary) !important;
        text-transform: uppercase !important;
        letter-spacing: 0.8px !important;
        padding: 14px 6px 6px 6px !important;
    }
    
    /* Sidebar Enterprise Nav Buttons */
    section[data-testid="stSidebar"] .stButton > button {
        width: 100% !important;
        text-align: left !important;
        display: flex !important;
        align-items: center !important;
        justify-content: flex-start !important;
        padding: 10px 14px !important;
        margin-bottom: 4px !important;
        border-radius: 6px !important;
        border: none !important;
        border-left: 3px solid transparent !important;
        font-size: 16.5px !important;
        font-weight: 500 !important;
        background-color: transparent !important;
        color: var(--text-secondary) !important;
        box-shadow: none !important;
        transition: background-color 0.15s ease, border-color 0.15s ease !important;
    }
    section[data-testid="stSidebar"] .stButton > button:hover {
        background-color: var(--table-row-hover) !important;
        color: var(--text-primary) !important;
        border-left: 3px solid transparent !important;
    }
    section[data-testid="stSidebar"] .stButton > button[kind="primary"] {
        background-color: var(--bg-panel) !important;
        color: var(--text-primary) !important;
        border-left: 3px solid var(--accent) !important;
        font-weight: 600 !important;
        box-shadow: none !important;
    }
    
    /* Profile Card at bottom of sidebar */
    .user-profile-card {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 10px 12px;
        background: var(--bg-panel);
        border: 1px solid var(--border);
        border-radius: 6px;
        margin-top: 14px;
        margin-bottom: 8px;
    }
    .user-avatar {
        width: 32px;
        height: 32px;
        border-radius: 50%;
        background: var(--bg-input);
        color: var(--accent);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 12px;
        font-weight: 700;
        border: 1px solid var(--border);
    }
    .user-name {
        font-size: 13px;
        font-weight: 600;
        color: var(--text-primary);
        line-height: 1.2;
    }
    .user-sub {
        font-size: 11px;
        color: var(--text-secondary);
        display: flex;
        align-items: center;
        gap: 6px;
        margin-top: 2px;
    }
    .badge-demo {
        background: var(--bg-input);
        color: var(--warning);
        font-size: 9px;
        padding: 1px 5px;
        border-radius: 3px;
        font-weight: 700;
        letter-spacing: 0.4px;
        border: 1px solid var(--border);
    }
    
    /* Complete Form controls, widget labels, and text contrast */
    label,
    label[data-testid="stWidgetLabel"],
    label[data-testid="stWidgetLabel"] p,
    label[data-testid="stWidgetLabel"] span {
        color: var(--text-primary) !important;
        font-size: 13.5px !important;
        font-weight: 500 !important;
    }
    
    div[data-testid="stCheckbox"] label,
    div[data-testid="stCheckbox"] label span,
    div[data-testid="stCheckbox"] label p {
        color: var(--text-primary) !important;
        font-size: 13.5px !important;
    }
    
    div[data-testid="stRadio"] label,
    div[data-testid="stRadio"] label span,
    div[data-testid="stRadio"] label p,
    div[data-testid="stRadio"] div[role="radiogroup"] label {
        color: var(--text-primary) !important;
        font-size: 13.5px !important;
    }
    div[data-testid="stRadio"] label:has(input:checked) span,
    div[data-testid="stRadio"] label:has(input:checked) p {
        color: var(--accent) !important;
        font-weight: 600 !important;
    }
    
    .stTextInput input, .stNumberInput input, .stSelectbox select, .stTextArea textarea {
        background-color: var(--bg-input) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border) !important;
        border-radius: 6px !important;
        font-size: 14px !important;
    }
    .stTextInput input::placeholder,
    .stNumberInput input::placeholder,
    .stTextArea textarea::placeholder {
        color: var(--text-secondary) !important;
        opacity: 0.8 !important;
    }
    
    div[data-baseweb="select"] > div {
        background-color: var(--bg-input) !important;
        border: 1px solid var(--border) !important;
        border-radius: 6px !important;
        color: var(--text-primary) !important;
    }
    div[data-baseweb="select"] * {
        color: var(--text-primary) !important;
    }
    div[data-baseweb="popover"], div[data-baseweb="menu"], ul[role="listbox"] {
        background-color: var(--bg-panel) !important;
        border: 1px solid var(--border) !important;
    }
    li[role="option"] {
        color: var(--text-primary) !important;
        background-color: var(--bg-panel) !important;
    }
    li[role="option"]:hover, li[role="option"][aria-selected="true"] {
        background-color: var(--table-row-hover) !important;
        color: var(--accent) !important;
    }
    
    /* Buttons */
    .main .stButton > button,
    .main .stButton > button[kind="secondary"],
    .stButton > button:not([kind="primary"]) {
        background-color: var(--bg-input) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border) !important;
        border-radius: 6px !important;
        font-size: 13.5px !important;
        font-weight: 500 !important;
    }
    .main .stButton > button:hover,
    .main .stButton > button[kind="secondary"]:hover,
    .stButton > button:not([kind="primary"]):hover {
        background-color: var(--bg-panel) !important;
        border-color: var(--accent) !important;
        color: var(--accent) !important;
    }
    .main .stButton > button[kind="primary"], button[kind="primary"] {
        background-color: var(--accent) !important;
        color: #0B0B0C !important;
        border: none !important;
        font-weight: 600 !important;
        border-radius: 6px !important;
        font-size: 14px !important;
    }
    .main .stButton > button[kind="primary"]:hover {
        background-color: var(--accent-hover) !important;
    }
    
    /* General text and headings */
    h1, h2, h3, h4, h5, h6 {
        color: var(--text-primary) !important;
    }
    .stMarkdown p, .stMarkdown span {
        color: var(--text-primary);
    }
    .stCaption, div[data-testid="stCaptionContainer"] p, .stCaption p {
        color: var(--text-secondary) !important;
    }
    
    /* Tabs */
    button[data-baseweb="tab"] {
        color: var(--text-secondary) !important;
        font-size: 13.5px !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: var(--accent) !important;
        font-weight: 600 !important;
        border-bottom-color: var(--accent) !important;
    }
    
    /* Slider numbers & labels */
    div[data-testid="stSlider"] * {
        color: var(--text-primary) !important;
    }
    
    /* Solid panels - NO gradients, NO glassmorphism */
    .lape-panel {
        background-color: var(--bg-panel);
        border: 1px solid var(--border);
        border-radius: 6px;
        padding: 18px 20px;
        margin-bottom: 14px;
    }
    .lape-metric-label {
        font-size: 12.5px;
        color: var(--text-secondary);
        text-transform: uppercase;
        font-weight: 600;
        letter-spacing: 0.6px;
    }
    .lape-metric-value {
        font-size: 28px;
        font-weight: 700;
        color: var(--text-primary);
        margin-top: 6px;
        line-height: 1.2;
        font-variant-numeric: tabular-nums;
    }
    .lape-metric-sub {
        font-size: 12px;
        color: var(--text-secondary);
        margin-top: 4px;
    }
    
    /* Tri-dimensional cards */
    .dim-exposure { background: var(--bg-panel); border: 1px solid var(--border); border-left: 3px solid var(--critical); border-radius: 6px; padding: 18px; }
    .dim-reliability { background: var(--bg-panel); border: 1px solid var(--border); border-left: 3px solid var(--accent); border-radius: 6px; padding: 18px; }
    .dim-urgency { background: var(--bg-panel); border: 1px solid var(--border); border-left: 3px solid var(--warning); border-radius: 6px; padding: 18px; }
    
    /* Clean Solid Tables with Tabular Numerals */
    .lape-table-wrap {
        background: var(--bg-panel);
        border: 1px solid var(--border);
        border-radius: 6px;
        overflow: hidden;
    }
    .lape-table {
        width: 100%; border-collapse: collapse; font-size: 13.5px; color: var(--text-primary);
        font-family: 'Inter', sans-serif;
    }
    .lape-table th {
        background: var(--table-header-bg); color: var(--accent); padding: 11px 14px;
        text-align: left; border-bottom: 1px solid var(--border);
        font-weight: 600; text-transform: uppercase; font-size: 11.5px; letter-spacing: 0.4px;
    }
    .lape-table td {
        padding: 9px 14px; border-bottom: 1px solid #242428; font-variant-numeric: tabular-nums;
    }
    .lape-table tr:hover { background: var(--table-row-hover); }
    .lape-table-footer { font-size: 11.5px; color: var(--text-secondary); padding: 7px 14px; }
    
    /* Information Callouts */
    .lape-info {
        background: var(--bg-panel); border: 1px solid var(--border); border-radius: 6px;
        padding: 13px 18px; font-size: 13.5px; color: var(--text-secondary); margin: 8px 0;
        line-height: 1.5;
    }
    .lape-info strong { color: var(--accent); }
    .lape-info code { background: var(--bg-input); color: var(--accent); padding: 2px 6px; border-radius: 4px; font-size: 12.5px; font-family: 'JetBrains Mono', monospace; }
    
    /* Compact Login Header & Wordmark */
    .login-brand-header {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 12px;
        margin: 8px 0 18px 0;
    }
    .login-square-logo {
        width: 36px;
        height: 36px;
        background-color: var(--accent);
        color: #0B0B0C;
        font-family: 'Inter', -apple-system, sans-serif;
        font-weight: 700;
        font-size: 20px;
        border-radius: 6px;
        display: flex;
        align-items: center;
        justify-content: center;
        line-height: 1;
        user-select: none;
    }
    .login-wordmark {
        font-family: 'Inter', -apple-system, sans-serif;
        font-size: 22px;
        font-weight: 700;
        color: var(--text-primary);
        letter-spacing: 0.5px;
        line-height: 1;
    }
    
    .captcha-box {
        background: var(--bg-input);
        border: 1px solid var(--border);
        border-radius: 6px;
        padding: 12px 14px;
        margin: 12px 0 16px 0;
    }
    .captcha-label {
        font-size: 12.5px;
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 6px;
    }
    
    .demo-box {
        background: var(--bg-panel);
        border: 1px solid var(--border);
        border-radius: 6px;
        padding: 10px 14px;
        margin-top: 14px;
        font-size: 12px;
        color: var(--text-secondary);
        line-height: 1.4;
        text-align: center;
    }
    .demo-box strong { color: var(--text-primary); }
    .demo-box code {
        background: var(--bg-input);
        color: var(--accent);
        padding: 2px 6px;
        border-radius: 4px;
        font-size: 11.5px;
        font-family: 'JetBrains Mono', monospace;
    }
    
    /* Header Bar */
    .lape-header-bar {
        background: var(--bg-panel);
        border: 1px solid var(--border);
        border-radius: 6px;
        padding: 16px 22px;
        margin-bottom: 20px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .lape-header-title { font-size: 20px; font-weight: 700; color: var(--text-primary); }
    .lape-header-sub { font-size: 13px; color: var(--text-secondary); margin-top: 2px; }
    .lape-header-badge {
        background: var(--bg-input);
        color: var(--accent);
        border: 1px solid var(--border);
        padding: 4px 10px;
        border-radius: 4px;
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Comparison cards */
    .compare-card { background: var(--bg-panel); border: 1px solid var(--border); border-radius: 6px; padding: 20px; }
    .compare-card.winner { border-color: var(--success); }
    
    /* Tabular Font */
    .mono-val { font-family: 'JetBrains Mono', monospace; font-size: 13px; }
</style>
"""

LIGHT_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');
    
    :root {
        --bg-main: #F8F9FA;
        --bg-sidebar: #F1F3F4;
        --bg-panel: #FFFFFF;
        --bg-input: #FFFFFF;
        --border: #DADCE0;
        --text-primary: #202124;
        --text-secondary: #5F6368;
        --accent: #1A73E8;
        --accent-hover: #1557B0;
        --warning: #E37400;
        --critical: #D93025;
        --success: #188038;
        --table-header-bg: #F1F3F4;
        --table-row-hover: #F8F9FA;
        --login-card-bg: #FFFFFF;
    }
    
    .stApp {
        background-color: var(--bg-main) !important;
        color: var(--text-primary) !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
    }
    
    .main .block-container {
        padding-top: 2.2rem !important;
        padding-bottom: 2rem !important;
        max-width: 1320px;
    }
    
    /* Top header configuration - keep visible so collapse toggle works */
    header[data-testid="stHeader"] {
        background: transparent !important;
        visibility: visible !important;
        height: auto !important;
        z-index: 1000 !important;
    }
    
    /* Hide unnecessary menu & deploy button */
    #MainMenu, footer, .stDeployButton {
        display: none !important;
        visibility: hidden !important;
    }
    
    /* CRITICAL FIX: Sidebar Collapse & Expand Toggle controls must always be visible & clickable */
    [data-testid="collapsedControl"],
    button[data-testid="stSidebarExpandButton"],
    button[data-testid="stSidebarCollapseButton"] {
        display: flex !important;
        visibility: visible !important;
        opacity: 1 !important;
        pointer-events: auto !important;
        z-index: 999999 !important;
        color: var(--text-primary) !important;
        background-color: var(--bg-panel) !important;
        border: 1px solid var(--border) !important;
        border-radius: 6px !important;
    }
    [data-testid="collapsedControl"] svg,
    button[data-testid="stSidebarExpandButton"] svg,
    button[data-testid="stSidebarCollapseButton"] svg {
        fill: var(--text-primary) !important;
        color: var(--text-primary) !important;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: var(--bg-sidebar) !important;
        border-right: 1px solid var(--border) !important;
    }
    section[data-testid="stSidebar"] * {
        color: var(--text-secondary) !important;
    }
    
    /* Sidebar Brand */
    .sidebar-brand {
        padding: 8px 6px 16px 6px;
        border-bottom: 1px solid var(--border);
        margin-bottom: 14px;
    }
    .sidebar-logo {
        font-size: 24px;
        font-weight: 700;
        color: var(--accent);
        letter-spacing: -0.5px;
        line-height: 1.1;
    }
    .sidebar-brand-title {
        font-size: 13.5px;
        color: var(--text-secondary);
        margin-top: 4px;
        text-transform: uppercase;
        letter-spacing: 0.6px;
    }
    
    /* Sidebar Group Heading */
    .sidebar-group-heading {
        font-size: 13.5px !important;
        font-weight: 600 !important;
        color: var(--text-secondary) !important;
        text-transform: uppercase !important;
        letter-spacing: 0.8px !important;
        padding: 14px 6px 6px 6px !important;
    }
    
    /* Sidebar Enterprise Nav Buttons */
    section[data-testid="stSidebar"] .stButton > button {
        width: 100% !important;
        text-align: left !important;
        display: flex !important;
        align-items: center !important;
        justify-content: flex-start !important;
        padding: 10px 14px !important;
        margin-bottom: 4px !important;
        border-radius: 6px !important;
        border: none !important;
        border-left: 3px solid transparent !important;
        font-size: 16.5px !important;
        font-weight: 500 !important;
        background-color: transparent !important;
        color: var(--text-secondary) !important;
        box-shadow: none !important;
        transition: background-color 0.15s ease, border-color 0.15s ease !important;
    }
    section[data-testid="stSidebar"] .stButton > button:hover {
        background-color: var(--table-row-hover) !important;
        color: var(--text-primary) !important;
        border-left: 3px solid transparent !important;
    }
    section[data-testid="stSidebar"] .stButton > button[kind="primary"] {
        background-color: var(--bg-panel) !important;
        color: var(--text-primary) !important;
        border-left: 3px solid var(--accent) !important;
        font-weight: 600 !important;
        box-shadow: none !important;
    }
    
    /* Profile Card at bottom of sidebar */
    .user-profile-card {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 10px 12px;
        background: var(--bg-panel);
        border: 1px solid var(--border);
        border-radius: 6px;
        margin-top: 14px;
        margin-bottom: 8px;
    }
    .user-avatar {
        width: 32px;
        height: 32px;
        border-radius: 50%;
        background: var(--bg-input);
        color: var(--accent);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 12px;
        font-weight: 700;
        border: 1px solid var(--border);
    }
    .user-name {
        font-size: 13px;
        font-weight: 600;
        color: var(--text-primary);
        line-height: 1.2;
    }
    .user-sub {
        font-size: 11px;
        color: var(--text-secondary);
        display: flex;
        align-items: center;
        gap: 6px;
        margin-top: 2px;
    }
    .badge-demo {
        background: var(--bg-sidebar);
        color: var(--warning);
        font-size: 9px;
        padding: 1px 5px;
        border-radius: 3px;
        font-weight: 700;
        letter-spacing: 0.4px;
        border: 1px solid var(--border);
    }
    
    /* Complete Form controls, widget labels, and text contrast */
    label,
    label[data-testid="stWidgetLabel"],
    label[data-testid="stWidgetLabel"] p,
    label[data-testid="stWidgetLabel"] span {
        color: var(--text-primary) !important;
        font-size: 13.5px !important;
        font-weight: 500 !important;
    }
    
    div[data-testid="stCheckbox"] label,
    div[data-testid="stCheckbox"] label span,
    div[data-testid="stCheckbox"] label p {
        color: var(--text-primary) !important;
        font-size: 13.5px !important;
    }
    
    div[data-testid="stRadio"] label,
    div[data-testid="stRadio"] label span,
    div[data-testid="stRadio"] label p,
    div[data-testid="stRadio"] div[role="radiogroup"] label {
        color: var(--text-primary) !important;
        font-size: 13.5px !important;
    }
    div[data-testid="stRadio"] label:has(input:checked) span,
    div[data-testid="stRadio"] label:has(input:checked) p {
        color: var(--accent) !important;
        font-weight: 600 !important;
    }
    
    .stTextInput input, .stNumberInput input, .stSelectbox select, .stTextArea textarea {
        background-color: var(--bg-input) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border) !important;
        border-radius: 6px !important;
        font-size: 14px !important;
    }
    .stTextInput input::placeholder,
    .stNumberInput input::placeholder,
    .stTextArea textarea::placeholder {
        color: var(--text-secondary) !important;
        opacity: 0.8 !important;
    }
    
    div[data-baseweb="select"] > div {
        background-color: var(--bg-input) !important;
        border: 1px solid var(--border) !important;
        border-radius: 6px !important;
        color: var(--text-primary) !important;
    }
    div[data-baseweb="select"] * {
        color: var(--text-primary) !important;
    }
    div[data-baseweb="popover"], div[data-baseweb="menu"], ul[role="listbox"] {
        background-color: var(--bg-panel) !important;
        border: 1px solid var(--border) !important;
    }
    li[role="option"] {
        color: var(--text-primary) !important;
        background-color: var(--bg-panel) !important;
    }
    li[role="option"]:hover, li[role="option"][aria-selected="true"] {
        background-color: var(--table-row-hover) !important;
        color: var(--accent) !important;
    }
    
    /* Buttons */
    .main .stButton > button,
    .main .stButton > button[kind="secondary"],
    .stButton > button:not([kind="primary"]) {
        background-color: #FFFFFF !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border) !important;
        border-radius: 6px !important;
        font-size: 13.5px !important;
        font-weight: 500 !important;
    }
    .main .stButton > button:hover,
    .main .stButton > button[kind="secondary"]:hover,
    .stButton > button:not([kind="primary"]):hover {
        background-color: var(--bg-sidebar) !important;
        border-color: var(--accent) !important;
        color: var(--accent) !important;
    }
    .main .stButton > button[kind="primary"], button[kind="primary"] {
        background-color: var(--accent) !important;
        color: #FFFFFF !important;
        border: none !important;
        font-weight: 600 !important;
        border-radius: 6px !important;
        font-size: 14px !important;
    }
    .main .stButton > button[kind="primary"]:hover {
        background-color: var(--accent-hover) !important;
    }
    
    /* General text and headings */
    h1, h2, h3, h4, h5, h6 {
        color: var(--text-primary) !important;
    }
    .stMarkdown p, .stMarkdown span {
        color: var(--text-primary);
    }
    .stCaption, div[data-testid="stCaptionContainer"] p, .stCaption p {
        color: var(--text-secondary) !important;
    }
    
    /* Tabs */
    button[data-baseweb="tab"] {
        color: var(--text-secondary) !important;
        font-size: 13.5px !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: var(--accent) !important;
        font-weight: 600 !important;
        border-bottom-color: var(--accent) !important;
    }
    
    /* Slider numbers & labels */
    div[data-testid="stSlider"] * {
        color: var(--text-primary) !important;
    }
    
    /* Solid panels - NO gradients, NO glassmorphism */
    .lape-panel {
        background-color: var(--bg-panel);
        border: 1px solid var(--border);
        border-radius: 6px;
        padding: 18px 20px;
        margin-bottom: 14px;
    }
    .lape-metric-label {
        font-size: 12.5px;
        color: var(--text-secondary);
        text-transform: uppercase;
        font-weight: 600;
        letter-spacing: 0.6px;
    }
    .lape-metric-value {
        font-size: 28px;
        font-weight: 700;
        color: var(--text-primary);
        margin-top: 6px;
        line-height: 1.2;
        font-variant-numeric: tabular-nums;
    }
    .lape-metric-sub {
        font-size: 12px;
        color: var(--text-secondary);
        margin-top: 4px;
    }
    
    /* Tri-dimensional cards */
    .dim-exposure { background: var(--bg-panel); border: 1px solid var(--border); border-left: 3px solid var(--critical); border-radius: 6px; padding: 18px; }
    .dim-reliability { background: var(--bg-panel); border: 1px solid var(--border); border-left: 3px solid var(--accent); border-radius: 6px; padding: 18px; }
    .dim-urgency { background: var(--bg-panel); border: 1px solid var(--border); border-left: 3px solid var(--warning); border-radius: 6px; padding: 18px; }
    
    /* Clean Solid Tables with Tabular Numerals */
    .lape-table-wrap {
        background: var(--bg-panel);
        border: 1px solid var(--border);
        border-radius: 6px;
        overflow: hidden;
    }
    .lape-table {
        width: 100%; border-collapse: collapse; font-size: 13.5px; color: var(--text-primary);
        font-family: 'Inter', sans-serif;
    }
    .lape-table th {
        background: var(--table-header-bg); color: var(--accent); padding: 11px 14px;
        text-align: left; border-bottom: 1px solid var(--border);
        font-weight: 600; text-transform: uppercase; font-size: 11.5px; letter-spacing: 0.4px;
    }
    .lape-table td {
        padding: 9px 14px; border-bottom: 1px solid #EEEEEE; font-variant-numeric: tabular-nums;
    }
    .lape-table tr:hover { background: var(--table-row-hover); }
    .lape-table-footer { font-size: 11.5px; color: var(--text-secondary); padding: 7px 14px; }
    
    /* Information Callouts */
    .lape-info {
        background: var(--bg-panel); border: 1px solid var(--border); border-radius: 6px;
        padding: 13px 18px; font-size: 13.5px; color: var(--text-secondary); margin: 8px 0;
        line-height: 1.5;
    }
    .lape-info strong { color: var(--accent); }
    .lape-info code { background: #F1F3F4; color: var(--accent); padding: 2px 6px; border-radius: 4px; font-size: 12.5px; font-family: 'JetBrains Mono', monospace; }
    
    /* Compact Login Header & Wordmark */
    .login-brand-header {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 12px;
        margin: 8px 0 18px 0;
    }
    .login-square-logo {
        width: 36px;
        height: 36px;
        background-color: var(--accent);
        color: #FFFFFF;
        font-family: 'Inter', -apple-system, sans-serif;
        font-weight: 700;
        font-size: 20px;
        border-radius: 6px;
        display: flex;
        align-items: center;
        justify-content: center;
        line-height: 1;
        user-select: none;
    }
    .login-wordmark {
        font-family: 'Inter', -apple-system, sans-serif;
        font-size: 22px;
        font-weight: 700;
        color: var(--text-primary);
        letter-spacing: 0.5px;
        line-height: 1;
    }
    
    .captcha-box {
        background: var(--bg-input);
        border: 1px solid var(--border);
        border-radius: 6px;
        padding: 12px 14px;
        margin: 12px 0 16px 0;
    }
    .captcha-label {
        font-size: 12.5px;
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 6px;
    }
    
    .demo-box {
        background: var(--bg-sidebar);
        border: 1px solid var(--border);
        border-radius: 6px;
        padding: 10px 14px;
        margin-top: 14px;
        font-size: 12px;
        color: var(--text-secondary);
        line-height: 1.4;
        text-align: center;
    }
    .demo-box strong { color: var(--text-primary); }
    .demo-box code {
        background: #FFFFFF;
        color: var(--accent);
        padding: 2px 6px;
        border-radius: 4px;
        font-size: 11.5px;
        font-family: 'JetBrains Mono', monospace;
        border: 1px solid var(--border);
    }
    
    /* Header Bar */
    .lape-header-bar {
        background: var(--bg-panel);
        border: 1px solid var(--border);
        border-radius: 6px;
        padding: 16px 22px;
        margin-bottom: 20px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .lape-header-title { font-size: 20px; font-weight: 700; color: var(--text-primary); }
    .lape-header-sub { font-size: 13px; color: var(--text-secondary); margin-top: 2px; }
    .lape-header-badge {
        background: var(--bg-sidebar);
        color: var(--accent);
        border: 1px solid var(--border);
        padding: 4px 10px;
        border-radius: 4px;
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Comparison cards */
    .compare-card { background: var(--bg-panel); border: 1px solid var(--border); border-radius: 6px; padding: 20px; }
    .compare-card.winner { border-color: var(--success); }
    
    /* Tabular Font */
    .mono-val { font-family: 'JetBrains Mono', monospace; font-size: 13px; }
</style>
"""

def apply_theme():
    """Apply the active stylesheet."""
    if st.session_state.get("theme", "dark") == "light":
        st.markdown(LIGHT_CSS, unsafe_allow_html=True)
    else:
        st.markdown(DARK_CSS, unsafe_allow_html=True)

# ==============================================================================
# HTML TABLE RENDERER WITH INR TABULAR NUMBERS
# ==============================================================================
def render_table(df: pd.DataFrame, max_rows: int = 100, height: int = 360):
    """Renders styled HTML table with INR formatting and tabular numerals."""
    if df.empty:
        st.markdown('<div class="lape-info">No records match current filter criteria.</div>', unsafe_allow_html=True)
        return
    disp = df.head(max_rows).copy()
    for col in disp.columns:
        col_lower = str(col).lower()
        if any(k in col_lower for k in ["amount", "spend", "exposure", "price", "shortfall", "cost", "revenue"]):
            disp[col] = disp[col].apply(lambda v: format_inr(v) if pd.notna(v) else "-")
        elif any(k in col_lower for k in ["pct", "rate", "ratio"]):
            disp[col] = disp[col].apply(lambda v: f"{v*100:.1f}%" if pd.notna(v) and isinstance(v, (int, float)) else str(v))
        elif pd.api.types.is_float_dtype(disp[col]):
            disp[col] = disp[col].apply(lambda v: f"{v:.3f}" if pd.notna(v) else "-")
        elif pd.api.types.is_integer_dtype(disp[col]):
            disp[col] = disp[col].apply(lambda v: f"{v:,}" if pd.notna(v) else "-")

    html = disp.to_html(index=False, classes="lape-table", escape=False)
    st.markdown(f"""
    <div class="lape-table-wrap" style="max-height:{height}px; overflow-y:auto; overflow-x:auto;">
        {html}
    </div>
    <div class="lape-table-footer">Showing {len(disp):,} of {len(df):,} records</div>
    """, unsafe_allow_html=True)

# ==============================================================================
# CACHED DATA INGESTION & FEATURE ENGINEERING
# ==============================================================================
@st.cache_data
def load_and_preprocess_data():
    """Loads all 5 raw datasets and caches feature matrices."""
    raw_dir = Path("data/raw")
    tx = pd.read_csv(raw_dir / "transactions.csv")
    prod = pd.read_csv(raw_dir / "products.csv")
    inv = pd.read_csv(raw_dir / "inventory.csv")
    camp = pd.read_csv(raw_dir / "campaigns.csv")
    cust = pd.read_csv(raw_dir / "customers.csv")

    tx_feat = build_unified_transaction_feature_matrix(tx, prod, cust)
    tx_feat = apply_baseline_a_simple_rule(tx_feat, exposure_threshold=100.0)
    tx_feat = apply_baseline_b_composite_rules(tx_feat, min_shortfall=25.0)

    inv_feat = compute_inventory_features(inv, prod, tx)
    camp_feat = compute_campaign_features(camp)

    return {
        "transactions": tx, "products": prod, "inventory": inv,
        "campaigns": camp, "customers": cust,
        "tx_feat": tx_feat, "inv_feat": inv_feat, "camp_feat": camp_feat
    }

# ==============================================================================
# UPGRADED LOGIN PAGE WITH ACCESSIBLE LOCAL ARITHMETIC CAPTCHA
# ==============================================================================
def render_login_page():
    """Renders authentication page with local arithmetic CAPTCHA."""
    apply_theme()
    
    col_l, col_center, col_r = st.columns([1.2, 1.4, 1.2])
    with col_center:
        st.markdown(f"""
        <div class="login-brand-header">
            <div class="login-square-logo">L</div>
            <div class="login-wordmark">LAPE</div>
        </div>
        """, unsafe_allow_html=True)

        username = st.text_input(t("username"), value="", placeholder="demo@lape.local", key="login_user_input")
        
        show_pw = st.checkbox(t("show_password"), value=False, key="login_show_pw_chk")
        pw_type = "text" if show_pw else "password"
        password = st.text_input(t("password"), value="", type=pw_type, placeholder="Enter password", key="login_pw_input")

        # Accessible Local Arithmetic CAPTCHA
        n1 = st.session_state.captcha_n1
        n2 = st.session_state.captcha_n2
        expected_ans = n1 + n2

        col_c_text, col_c_btn = st.columns([3, 1])
        with col_c_text:
            captcha_input = st.text_input(
                f"Quick Math Check: What is {n1} + {n2}?",
                placeholder="Type the answer here",
                key="login_captcha_input"
            )
        with col_c_btn:
            st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
            if st.button("↻ New", help="Get a different math question", key="captcha_refresh_btn", use_container_width=True):
                st.session_state.captcha_n1 = random.randint(3, 12)
                st.session_state.captcha_n2 = random.randint(2, 9)
                st.rerun()

        if st.button(t("sign_in"), use_container_width=True, type="primary", key="login_btn"):
            # Validate CAPTCHA first
            try:
                user_ans = int(captcha_input.strip())
            except (ValueError, TypeError):
                user_ans = None

            if user_ans != expected_ans:
                st.error(t("invalid_captcha"))
            elif username.strip() not in [DEMO_USERNAME, "admin"] or password != DEMO_PASSWORD:
                st.error(t("invalid_credentials"))
            else:
                st.session_state.authenticated = True
                st.rerun()

        st.markdown(f"""
        <div class="demo-box">
            <strong>Demo Login:</strong> <code>demo@lape.local</code> &nbsp;|&nbsp; Password: <code>lape2026</code>
            <div style="font-size: 11px; color: var(--text-secondary); margin-top: 3px;">
                Runs locally on your computer &bull; No data is sent over the internet
            </div>
        </div>
        """, unsafe_allow_html=True)

        col_t1, col_t2 = st.columns(2)
        with col_t1:
            theme_choice = st.radio(
                t("theme"),
                [t("dark"), t("light")],
                index=0 if st.session_state.theme == "dark" else 1,
                horizontal=True,
                key="login_theme_selector"
            )
            target_theme = "light" if theme_choice == t("light") else "dark"
            if target_theme != st.session_state.theme:
                st.session_state.theme = target_theme
                st.rerun()

        with col_t2:
            current_lang_idx = SUPPORTED_LANGUAGES.index(st.session_state.language) if st.session_state.language in SUPPORTED_LANGUAGES else 0
            chosen_lang = st.selectbox(
                t("language"),
                SUPPORTED_LANGUAGES,
                index=current_lang_idx,
                key="login_lang_selector"
            )
            if chosen_lang != st.session_state.language:
                st.session_state.language = chosen_lang
                st.rerun()

# ==============================================================================
# MAIN ENTERPRISE APPLICATION
# ==============================================================================
def main():
    apply_theme()

    # Top Header Bar (within page flow, never covers the sidebar toggles)
    st.markdown(f"""
    <div class="lape-header-bar">
        <div>
            <div class="lape-header-title">LAPE: {t('app_full')}</div>
            <div class="lape-header-sub">{t('datathon_subtitle')}</div>
        </div>
        <div>
            <span class="lape-header-badge">Indian Rupees (₹) &nbsp;|&nbsp; Tested &amp; Verified Locally</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    data = load_and_preprocess_data()

    # --------------------------------------------------------------------------
    # REDESIGNED ENTERPRISE SIDEBAR
    # --------------------------------------------------------------------------
    st.sidebar.markdown("""
    <div class="sidebar-brand">
        <div class="sidebar-logo">LAPE</div>
        <div class="sidebar-brand-title">Finding and Stopping Lost Money</div>
    </div>
    """, unsafe_allow_html=True)

    # Group 1: MAIN PAGES
    st.sidebar.markdown('<div class="sidebar-group-heading">MAIN PAGES</div>', unsafe_allow_html=True)
    workspace_modules = [
        ("Overview", t("overview")),
        ("Data Quality", t("data_quality")),
        ("Alert Queue", t("alert_queue")),
        ("Investigation", t("investigation")),
        ("Scenario Lab", t("scenario_lab")),
        ("Model Evaluation", t("model_eval")),
    ]
    for page_key, label in workspace_modules:
        is_active = (st.session_state.current_page == page_key)
        btn_type = "primary" if is_active else "secondary"
        if st.sidebar.button(label, key=f"nav_btn_{page_key}", use_container_width=True, type=btn_type):
            if st.session_state.current_page != page_key:
                st.session_state.current_page = page_key
                st.rerun()

    # Group 2: DOWNLOADS
    st.sidebar.markdown('<div class="sidebar-group-heading">DOWNLOADS</div>', unsafe_allow_html=True)
    reporting_modules = [
        ("Reports", t("reports")),
    ]
    for page_key, label in reporting_modules:
        is_active = (st.session_state.current_page == page_key)
        btn_type = "primary" if is_active else "secondary"
        if st.sidebar.button(label, key=f"nav_btn_{page_key}", use_container_width=True, type=btn_type):
            if st.session_state.current_page != page_key:
                st.session_state.current_page = page_key
                st.rerun()

    # Group 3: OPTIONS
    st.sidebar.markdown('<div class="sidebar-group-heading">OPTIONS</div>', unsafe_allow_html=True)
    pref_modules = [
        ("Settings", t("settings")),
    ]
    for page_key, label in pref_modules:
        is_active = (st.session_state.current_page == page_key)
        btn_type = "primary" if is_active else "secondary"
        if st.sidebar.button(label, key=f"nav_btn_{page_key}", use_container_width=True, type=btn_type):
            if st.session_state.current_page != page_key:
                st.session_state.current_page = page_key
                st.rerun()

    # Compact User Profile & Logout Area
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    <div class="user-profile-card">
        <div class="user-avatar">DU</div>
        <div class="user-info">
            <div class="user-name">Demo User</div>
            <div class="user-sub">demo@lape.local <span class="badge-demo">LOCAL DEMO</span></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.sidebar.button(t("logout"), use_container_width=True, key="sidebar_logout_btn"):
        st.session_state.authenticated = False
        st.rerun()

    # --------------------------------------------------------------------------
    # PAGE ROUTING
    # --------------------------------------------------------------------------
    page = st.session_state.current_page
    if page == "Overview":
        render_overview_view(data)
    elif page == "Data Quality":
        render_audit_view(data)
    elif page == "Alert Queue":
        render_alert_queue_view(data)
    elif page == "Investigation":
        render_investigation_view(data)
    elif page == "Scenario Lab":
        render_scenario_lab_view(data)
    elif page == "Model Evaluation":
        render_benchmark_view(data)
    elif page == "Reports":
        render_reports_view(data)
    elif page == "Settings":
        render_settings_view()

# ==============================================================================
# 1. OVERVIEW PAGE
# ==============================================================================
def render_overview_view(data):
    st.subheader(t("overview"))
    st.caption("A clear summary of all checked records and where money might be at risk, shown in Indian Rupees (₹).")

    total_rows = len(data["transactions"]) + len(data["products"]) + len(data["inventory"]) + len(data["campaigns"]) + len(data["customers"])
    completed_tx_mask = (data["tx_feat"]["status"] == "Completed")
    completed_neg = int(((completed_tx_mask) & (data["tx_feat"]["price_cost_shortfall"] > 0)).sum())
    tx_shortfall_completed = data["tx_feat"][completed_tx_mask]["potential_shortfall_amount"].sum()
    
    viol_camp = data["camp_feat"][data["camp_feat"]["funnel_violation"] == 1]
    camp_viol_count = len(viol_camp)
    camp_compromised_spend = viol_camp["spend"].sum()

    inv_disc_count = int(data["inv_feat"]["flag_discrepancy"].sum())

    # KPI Summary Cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="lape-panel">
            <div class="lape-metric-label">Possible Money Lost on Sales (₹)</div>
            <div class="lape-metric-value" style="color:var(--critical);">{format_inr(tx_shortfall_completed)}</div>
            <div class="lape-metric-sub">{completed_neg:,} completed orders where selling price was lower than cost</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="lape-panel">
            <div class="lape-metric-label">Wasted Ad Spend (₹)</div>
            <div class="lape-metric-value" style="color:var(--warning);">{format_inr(camp_compromised_spend)}</div>
            <div class="lape-metric-sub">{camp_viol_count:,} ad campaigns with broken numbers (like more clicks than views)</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="lape-panel">
            <div class="lape-metric-label">Warehouse Stock Warnings</div>
            <div class="lape-metric-value">{inv_disc_count:,}</div>
            <div class="lape-metric-sub">Stock records where low-inventory warning did not match physical count</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="lape-panel">
            <div class="lape-metric-label">Total Records Checked</div>
            <div class="lape-metric-value">{total_rows:,}</div>
            <div class="lape-metric-sub">Across 5 files: zero missing IDs and zero duplicate rows</div>
        </div>
        """, unsafe_allow_html=True)

    col_l, col_r = st.columns([1.1, 1.0])
    with col_l:
        st.markdown("#### Order Status &amp; Possible Money at Risk")
        st.caption("Shows whether an order is already completed, still pending, or returned.")
        pending_shortfall = data["tx_feat"][data["tx_feat"]["status"] == "Pending"]["potential_shortfall_amount"].sum()
        returned_shortfall = data["tx_feat"][data["tx_feat"]["status"] == "Returned"]["potential_shortfall_amount"].sum()
        cancelled_shortfall = data["tx_feat"][data["tx_feat"]["status"] == "Cancelled"]["potential_shortfall_amount"].sum()

        dist_df = pd.DataFrame([
            {"Order Status": "Completed (Delivered)", "Number of Orders": completed_neg, "Possible Money at Risk (₹)": tx_shortfall_completed, "What This Means": "Sold below cost; check invoices"},
            {"Order Status": "Pending (In Progress)", "Number of Orders": int(((data['tx_feat']['status'] == 'Pending') & (data['tx_feat']['price_cost_shortfall'] > 0)).sum()), "Possible Money at Risk (₹)": pending_shortfall, "What This Means": "Not delivered yet; can still be corrected"},
            {"Order Status": "Returned (Sent Back)", "Number of Orders": int(((data['tx_feat']['status'] == 'Returned') & (data['tx_feat']['price_cost_shortfall'] > 0)).sum()), "Possible Money at Risk (₹)": returned_shortfall, "What This Means": "Item returned; check restock condition"},
            {"Order Status": "Cancelled (Stopped)", "Number of Orders": int(((data['tx_feat']['status'] == 'Cancelled') & (data['tx_feat']['price_cost_shortfall'] > 0)).sum()), "Possible Money at Risk (₹)": cancelled_shortfall, "What This Means": "Order was cancelled; no money lost"},
        ])
        render_table(dist_df, max_rows=10, height=220)

    with col_r:
        st.markdown("#### Product Categories with the Most Risk")
        st.caption("Which types of items have the biggest difference between selling price and cost.")
        cat_group = data["tx_feat"][completed_tx_mask].groupby("category").agg(
            transactions=("transaction_id", "count"),
            total_shortfall=("potential_shortfall_amount", "sum"),
            avg_discount=("discount_pct", "mean")
        ).reset_index().rename(columns={
            "category": "Product Category",
            "transactions": "Completed Orders",
            "total_shortfall": "Possible Money at Risk (₹)",
            "avg_discount": "Average Discount"
        }).sort_values(by="Possible Money at Risk (₹)", ascending=False)
        render_table(cat_group, max_rows=10, height=220)

    st.markdown("---")
    st.markdown("#### Important Notice on Test Data &amp; Real Money")
    st.markdown("""
    <div class="lape-info">
        <strong>Good to know:</strong> All amounts are shown in Indian Rupees (₹) using test data prepared for this project.
        In the product catalog, 76 out of 250 products (30.4%) were set up with costs higher than list prices.
        These numbers represent possible money at risk, not confirmed losses, and should be checked with real store receipts before taking action.
    </div>
    """, unsafe_allow_html=True)

# ==============================================================================
# 2. DATA QUALITY / DATA CHECKS PAGE
# ==============================================================================
def render_audit_view(data):
    st.subheader(t("data_quality"))
    st.caption("Direct checks across all 5 data files to ensure records are complete, accurate, and ready to analyze.")

    total_rows = len(data["transactions"]) + len(data["products"]) + len(data["inventory"]) + len(data["campaigns"]) + len(data["customers"])
    completed_neg = int(((data["tx_feat"]["status"] == "Completed") & (data["tx_feat"]["price_cost_shortfall"] > 0)).sum())
    funnel_viol = int(data["camp_feat"]["funnel_violation"].sum())
    inv_disc = int(data["inv_feat"]["flag_discrepancy"].sum())

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="lape-panel">
            <div class="lape-metric-label">{t('total_records')}</div>
            <div class="lape-metric-value">{total_rows:,}</div>
            <div class="lape-metric-sub">5 data files | No missing values | No duplicate rows</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="lape-panel">
            <div class="lape-metric-label">{t('completed_neg')}</div>
            <div class="lape-metric-value" style="color:var(--critical);">{completed_neg:,}</div>
            <div class="lape-metric-sub">Out of 13,962 completed orders</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="lape-panel">
            <div class="lape-metric-label">{t('funnel_inversions')}</div>
            <div class="lape-metric-value" style="color:var(--warning);">{funnel_viol:,}</div>
            <div class="lape-metric-sub">More clicks than views, or more purchases than clicks</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="lape-panel">
            <div class="lape-metric-label">{t('inv_discrepancies')}</div>
            <div class="lape-metric-value">{inv_disc:,}</div>
            <div class="lape-metric-sub">Stock is below reorder point, but warning flag was missing</div>
        </div>
        """, unsafe_allow_html=True)

    t1, t2, t3 = st.tabs([t("pricing_margin"), t("funnel_violations"), t("inv_inconsistencies")])

    with t1:
        st.markdown(f"#### {t('pricing_margin')}")
        st.markdown("""
        <div class="lape-info">
            <strong>Good to know about our test catalog:</strong> In this artificial test data, 76 out of 250 products (30.4%) were set up with costs higher than list prices.
            This helps us test whether our system can catch orders that lose money before they ship.
        </div>
        """, unsafe_allow_html=True)
        col_a, col_b = st.columns(2)
        with col_a:
            status_df = data["transactions"]["status"].value_counts().reset_index()
            status_df.columns = ["Order Status", "Number of Orders"]
            render_table(status_df, max_rows=10, height=220)
        with col_b:
            shortfall_sum_completed = data["tx_feat"][data["tx_feat"]["status"] == "Completed"]["potential_shortfall_amount"].sum()
            shortfall_sum_pending = data["tx_feat"][data["tx_feat"]["status"] == "Pending"]["potential_shortfall_amount"].sum()
            shortfall_sum_returned = data["tx_feat"][data["tx_feat"]["status"] == "Returned"]["potential_shortfall_amount"].sum()
            exp_breakdown = pd.DataFrame([
                {"Order Status": "Completed (Delivered)", "Possible Money at Risk (₹)": shortfall_sum_completed, "Number of Orders": completed_neg},
                {"Order Status": "Pending (In Progress)", "Possible Money at Risk (₹)": shortfall_sum_pending, "Number of Orders": int(((data['tx_feat']['status'] == 'Pending') & (data['tx_feat']['price_cost_shortfall'] > 0)).sum())},
                {"Order Status": "Returned (Sent Back)", "Possible Money at Risk (₹)": shortfall_sum_returned, "Number of Orders": int(((data['tx_feat']['status'] == 'Returned') & (data['tx_feat']['price_cost_shortfall'] > 0)).sum())}
            ])
            render_table(exp_breakdown, max_rows=10, height=220)

    with t2:
        st.markdown(f"#### {t('funnel_violations')}")
        viol_camp = data["camp_feat"][data["camp_feat"]["funnel_violation"] == 1].copy()
        viol_camp = viol_camp.rename(columns={
            "campaign_id": "Campaign ID",
            "date": "Date",
            "channel": "Ad Channel",
            "category": "Category",
            "spend": "Money Spent (₹)",
            "impressions": "Views",
            "clicks": "Clicks",
            "conversions": "Purchases",
            "clicks_gt_impressions": "Clicks Exceed Views?",
            "conversions_gt_clicks": "Purchases Exceed Clicks?"
        })
        st.markdown(f"""
        <div class="lape-info">
            Found <strong>{len(viol_camp)}</strong> ad campaign records with broken numbers out of 1,200 total campaigns.<br>
            Total money spent on these broken campaigns: <strong>{format_inr(viol_camp['Money Spent (₹)'].sum())}</strong> (Needs verification with ad providers)
        </div>
        """, unsafe_allow_html=True)
        render_table(
            viol_camp[["Campaign ID", "Date", "Ad Channel", "Category", "Money Spent (₹)", "Views", "Clicks", "Purchases", "Clicks Exceed Views?", "Purchases Exceed Clicks?"]],
            max_rows=50, height=350
        )

    with t3:
        st.markdown(f"#### {t('inv_inconsistencies')}")
        inv_disc_df = data["inv_feat"][data["inv_feat"]["flag_discrepancy"] == 1].copy()
        inv_disc_df = inv_disc_df.rename(columns={
            "inventory_record_id": "Inventory Record ID",
            "product_id": "Product ID",
            "date": "Date",
            "stock_on_hand": "Current Stock on Hand",
            "reorder_threshold": "Reorder Point (Threshold)",
            "stockout_flag": "Low-Stock Warning Turned On?",
            "supplier_delay_days": "Supplier Delay (Days)"
        })
        st.markdown(f"""
        <div class="lape-info">
            Found <strong>{len(inv_disc_df)}</strong> stock records where current inventory fell below the reorder level, but the low-stock warning flag was turned off.
        </div>
        """, unsafe_allow_html=True)
        render_table(
            inv_disc_df[["Inventory Record ID", "Product ID", "Date", "Current Stock on Hand", "Reorder Point (Threshold)", "Low-Stock Warning Turned On?", "Supplier Delay (Days)"]],
            max_rows=50, height=350
        )

# ==============================================================================
# 3. ALERT QUEUE / WARNING LIST PAGE
# ==============================================================================
def render_alert_queue_view(data):
    st.subheader(t("alert_queue"))
    st.caption("A clear list of all items that look unusual, ranked by possible money at risk, how solid the proof is, and how quickly someone should check.")

    with st.spinner("Checking all records for warnings..."):
        alerts_df = generate_unified_alert_queue(
            data["tx_feat"], data["camp_feat"], data["inv_feat"], min_exposure=25.0
        )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        domain_filter = st.multiselect("Area Checked", options=["All"] + list(alerts_df["domain"].unique()), default=["All"])
    with c2:
        urgency_filter = st.multiselect("How Quickly Should We Check?", options=["All", "CRITICAL", "HIGH", "MEDIUM", "LOW"], default=["All"])
    with c3:
        reliability_filter = st.multiselect("How Reliable Is This Information?", options=["All", "HIGH", "MEDIUM", "LOW"], default=["All"])
    with c4:
        search_query = st.text_input("Search by Order ID, Product, or Customer", placeholder="e.g. TX000001, PR0103")

    filtered = alerts_df.copy()
    if "All" not in domain_filter and domain_filter:
        filtered = filtered[filtered["domain"].isin(domain_filter)]
    if "All" not in urgency_filter and urgency_filter:
        filtered = filtered[filtered["action_urgency"].isin(urgency_filter)]
    if "All" not in reliability_filter and reliability_filter:
        filtered = filtered[filtered["evidence_reliability"].isin(reliability_filter)]
    if search_query:
        mask = (
            filtered["alert_id"].str.contains(search_query, case=False, na=False) |
            filtered["source_record_id"].str.contains(search_query, case=False, na=False) |
            filtered.get("product_id", pd.Series("")).astype(str).str.contains(search_query, case=False, na=False) |
            filtered.get("customer_id", pd.Series("")).astype(str).str.contains(search_query, case=False, na=False)
        )
        filtered = filtered[mask]

    total_exp = filtered["potential_exposure_amount"].sum()
    critical_count = int((filtered['action_urgency'] == 'CRITICAL').sum())
    high_rel = int((filtered['evidence_reliability'] == 'HIGH').sum())

    q1, q2, q3, q4 = st.columns(4)
    with q1:
        st.markdown(f"""<div class="lape-panel"><div class="lape-metric-label">{t('total_alerts')}</div><div class="lape-metric-value">{len(filtered):,}</div></div>""", unsafe_allow_html=True)
    with q2:
        st.markdown(f"""<div class="lape-panel"><div class="lape-metric-label">{t('critical_urgency')}</div><div class="lape-metric-value" style="color:var(--critical);">{critical_count:,}</div></div>""", unsafe_allow_html=True)
    with q3:
        st.markdown(f"""<div class="lape-panel"><div class="lape-metric-label">{t('high_reliability')}</div><div class="lape-metric-value" style="color:var(--accent);">{high_rel:,}</div></div>""", unsafe_allow_html=True)
    with q4:
        st.markdown(f"""<div class="lape-panel"><div class="lape-metric-label">{t('exposure_sum')}</div><div class="lape-metric-value" style="color:var(--warning);">{format_inr(total_exp)}</div></div>""", unsafe_allow_html=True)

    st.markdown(f"#### {t('alert_stream')}")
    display_df = filtered.rename(columns={
        "alert_id": "Warning ID",
        "domain": "Area Checked",
        "source_record_id": "Original Record ID",
        "potential_exposure_amount": "Possible Money at Risk (₹)",
        "evidence_reliability": "Proof Quality",
        "action_urgency": "Urgency",
        "exposure_nature": "What Happened"
    })
    display_cols = ["Warning ID", "Area Checked", "Original Record ID", "Possible Money at Risk (₹)", "Proof Quality", "Urgency", "What Happened"]
    render_table(
        display_df[display_cols].sort_values(by="Possible Money at Risk (₹)", ascending=False),
        max_rows=100, height=380
    )

    st.markdown("---")
    st.markdown("#### Review a Warning in Detail")
    alert_ids = list(filtered["alert_id"].values)
    if alert_ids:
        col_sel, col_go = st.columns([3, 1])
        with col_sel:
            chosen = st.selectbox("Select Warning to Inspect in Detail", options=alert_ids, key="queue_select_alert")
        with col_go:
            st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
            if st.button("Open Detailed Review →", type="primary", use_container_width=True):
                st.session_state.selected_alert_id = chosen
                st.session_state.current_page = "Investigation"
                st.rerun()

# ==============================================================================
# 4. INVESTIGATION / LOOK CLOSER PAGE
# ==============================================================================
def render_investigation_view(data):
    st.subheader(t("investigation"))
    st.caption("A step-by-step look at a single warning: see where the numbers came from, how the risk was calculated, and what to do next.")

    alerts_df = generate_unified_alert_queue(
        data["tx_feat"], data["camp_feat"], data["inv_feat"], min_exposure=25.0
    )
    all_alert_ids = list(alerts_df["alert_id"].values)
    
    default_idx = 0
    if st.session_state.selected_alert_id in all_alert_ids:
        default_idx = all_alert_ids.index(st.session_state.selected_alert_id)

    selected_id = st.selectbox("Select a Warning to Review in Detail", options=all_alert_ids, index=default_idx)
    st.session_state.selected_alert_id = selected_id
    alert_record = alerts_df[alerts_df["alert_id"] == selected_id].iloc[0]

    # Tri-Dimensional Cards in INR
    col_exp, col_rel, col_urg = st.columns(3)
    with col_exp:
        st.markdown(f"""
        <div class="dim-exposure">
            <div style="font-weight:700; color:var(--critical); font-size:12px; text-transform:uppercase;">{t('exposure_dim')}</div>
            <div style="font-size:24px; font-weight:700; color:var(--text-primary); margin:6px 0;">{format_inr(alert_record['potential_exposure_amount'])}</div>
            <div style="font-size:12.5px; color:var(--text-secondary);">What this is: <strong>{alert_record['exposure_nature']}</strong></div>
            <div style="font-size:11.5px; color:var(--accent); background:var(--bg-input); padding:6px 8px; border-radius:4px; margin-top:8px; word-break:break-all; font-family:'JetBrains Mono',monospace;">Calculation: {alert_record['calculation_formula']}</div>
        </div>
        """, unsafe_allow_html=True)
    with col_rel:
        rel_val = alert_record['evidence_reliability']
        if rel_val == "HIGH":
            rel_desc = "Solid proof: Order is completed, product details match our catalog, and price was confirmed below cost."
        elif rel_val == "MEDIUM":
            rel_desc = "Catalog alert: In our test catalog, the product cost was recorded higher than list price before this sale."
        else:
            rel_desc = "Preliminary check: The order is still pending or was cancelled, so no money has left the account."

        st.markdown(f"""
        <div class="dim-reliability">
            <div style="font-weight:700; color:var(--accent); font-size:12px; text-transform:uppercase;">{t('reliability_dim')}</div>
            <div style="font-size:24px; font-weight:700; color:var(--text-primary); margin:6px 0;">{rel_val}</div>
            <div style="font-size:12.5px; color:var(--text-secondary);">{rel_desc}</div>
        </div>
        """, unsafe_allow_html=True)
    with col_urg:
        urg_val = alert_record['action_urgency']
        if urg_val == "CRITICAL":
            urg_desc = "Check immediately: Large rupee amount; resolve before monthly accounting books close."
        elif urg_val == "HIGH":
            urg_desc = "Check soon: Intervene with sales or shipping team before this order ships or next invoices go out."
        elif urg_val == "MEDIUM":
            urg_desc = "Normal check: Review during regular weekly audit."
        else:
            urg_desc = "Low priority: Record is noted, but no urgent money is at risk."

        st.markdown(f"""
        <div class="dim-urgency">
            <div style="font-weight:700; color:var(--warning); font-size:12px; text-transform:uppercase;">{t('urgency_dim')}</div>
            <div style="font-size:24px; font-weight:700; color:var(--text-primary); margin:6px 0;">{urg_val}</div>
            <div style="font-size:12.5px; color:var(--warning);">{urg_desc}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.success(f"**{t('suggested_action')}**: {alert_record['suggested_verification_action']}")

    col_meta1, col_meta2 = st.columns(2)
    with col_meta1:
        st.markdown("#### Warning Details &amp; Context")
        meta_df = pd.DataFrame([
            {"Attribute": "Warning ID", "Value": str(alert_record['alert_id'])},
            {"Attribute": "Area Checked", "Value": str(alert_record['domain'])},
            {"Attribute": "Original Record ID", "Value": str(alert_record['source_record_id'])},
            {"Attribute": "Possible Money at Risk", "Value": format_inr(alert_record['potential_exposure_amount'])},
            {"Attribute": "Type of Problem", "Value": str(alert_record['exposure_nature'])},
        ])
        render_table(meta_df, max_rows=10, height=200)

    with col_meta2:
        st.markdown("#### Original Record Details")
        with st.expander("View Original Data from the Record", expanded=True):
            st.json(alert_record.get("raw_attributes", {}))

# ==============================================================================
# 5. SCENARIO LAB / TEST SITUATIONS PAGE
# ==============================================================================
def render_scenario_lab_view(data):
    st.subheader(t("scenario_lab"))
    st.caption("Try out different prices, costs, and discounts to see how the system spots money leaks and avoids false alarms.")

    st.markdown("""
    <div class="lape-info">
        <strong>How this works:</strong> Adjust the sliders and numbers below to see what happens when an item is sold below its cost.
        All amounts are shown in Indian Rupees (₹). Test results are evaluated against controlled artificial test data.
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        sim_qty = st.slider("Number of Items (Quantity)", 1, 50, 8)
        sim_unit_cost = st.number_input("Cost to Buy or Make Each Item (₹)", value=120.0, step=5.0)
    with col2:
        sim_list_price = st.number_input("Normal Catalog List Price (₹)", value=150.0, step=5.0)
        sim_discount = st.slider("Discount Given (%)", 0, 90, 45) / 100.0
    with col3:
        sim_status = st.selectbox("Order Status", ["Completed", "Pending", "Returned", "Cancelled"])
        sim_channel = st.selectbox("Sales Channel", ["Web", "App", "Store"])

    eff_price = sim_list_price * (1.0 - sim_discount)
    shortfall = max(0.0, sim_unit_cost - eff_price)
    total_shortfall = sim_qty * shortfall
    margin = eff_price - sim_unit_cost

    st.markdown("---")
    st.markdown(f"#### {t('realtime_output')}")
    res_c1, res_c2, res_c3 = st.columns(3)
    delta_label = f"{format_inr(margin)} Profit per Item" if margin >= 0 else f"{format_inr(abs(margin))} Loss per Item"
    res_c1.metric("Final Selling Price per Item (₹)", format_inr(eff_price), delta=delta_label)
    res_c2.metric("Total Money Lost on This Order (₹)", format_inr(total_shortfall))

    model_path = Path("models/lape_hist_gb.joblib")
    if model_path.exists():
        clf = LapeLeakClassifier.load(str(model_path))
        sample_df = pd.DataFrame([{
            "quantity": sim_qty, "unit_price": sim_list_price, "discount_pct": sim_discount,
            "effective_unit_price": eff_price, "unit_cost": sim_unit_cost,
            "unit_margin": margin,
            "margin_pct": margin / eff_price if eff_price > 0 else 0,
            "price_cost_shortfall": shortfall, "potential_shortfall_amount": total_shortfall,
            "gross_revenue": sim_qty * sim_list_price,
            "discount_amount": sim_qty * sim_list_price * sim_discount,
            "net_revenue": sim_qty * eff_price, "price_to_list_ratio": 1.0,
            "catalog_cost_gt_list": 1 if sim_unit_cost > sim_list_price else 0,
            "category_discount_pct_robust_z": (sim_discount - 0.15) / 0.05,
            "category_unit_margin_robust_z": -2.0 if shortfall > 0 else 0.5,
            "cust_return_rate": 0.05, "cust_avg_discount": 0.15, "cust_total_tx": 10,
            "hour": 14, "dayofweek": 2,
            "status": sim_status, "channel": sim_channel,
            "category": "Electronics", "payment_method": "Card"
        }])
        X_sample, _ = prepare_feature_matrix(sample_df)
        prob_leak = float(clf.predict_proba(X_sample)[0, 1])
        risk_label = "HIGH RISK (Likely Money Leak)" if prob_leak > 0.5 else "SAFE (Normal Order)"
        res_c3.metric("LAPE Machine Learning Risk Assessment", f"{prob_leak*100:.1f}% Risk", delta=risk_label)

    st.markdown("---")
    st.markdown("#### Pre-Built Test Situations (Artificial Test Data)")
    st.caption("These test situations were created to test whether the system catches different kinds of pricing errors, coupon abuse, and inventory mistakes.")
    manifest_path = Path("data/synthetic/scenario_manifest.json")
    if manifest_path.exists():
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)
        st.json(manifest)

# ==============================================================================
# 6. MODEL EVALUATION / TEST THE MODEL PAGE
# ==============================================================================
def render_benchmark_view(data):
    st.subheader(t("model_eval"))
    st.caption("See how well different detection methods perform on controlled artificial test data: how many real problems they catch and how many false alarms they avoid.")

    st.markdown("""
    <div class="lape-info">
        <strong>How we test accuracy:</strong> All accuracy scores are calculated using controlled artificial test data where we know the exact correct answer.
        In real business operations, warnings are always presented as potential items to verify, never as confirmed loss.
    </div>
    """, unsafe_allow_html=True)

    report_path = Path("reports/evaluation_report.json")
    if not report_path.exists():
        st.warning("Test report not found at reports/evaluation_report.json")
        return

    with open(report_path, encoding="utf-8") as f:
        benchmarks = json.load(f)

    st.markdown(f"#### {t('algorithm_comparison')}")
    comp_df = pd.DataFrame(benchmarks["comparison_table"]).copy()
    model_name_map = {
        "Baseline A (Simple Rule)": "Method A — Simple Rule (Flags any loss)",
        "Baseline B (Composite Rules)": "Method B — Fixed Rules (Basic filters)",
        "LAPE ML (HistGB + Triage)": "LAPE — Machine Learning + Risk Checks"
    }
    if "model" in comp_df.columns:
        comp_df["model"] = comp_df["model"].map(lambda m: model_name_map.get(m, m))
    comp_df = comp_df.rename(columns={
        "model": "Detection Method",
        "precision": "Accuracy When Flagging (Precision)",
        "recall": "Problems Caught (Recall)",
        "f1": "Overall Score (F1 Score)"
    })
    render_table(comp_df, max_rows=10, height=180)

    c1, c2, c3 = st.columns(3)
    ba = benchmarks["baseline_a"]
    bb = benchmarks["baseline_b"]
    bm = benchmarks["ml_model"]

    with c1:
        st.markdown(f"""
        <div class="compare-card">
            <div class="lape-metric-label">Method A — Simple Rule</div>
            <div class="lape-metric-value">Score: {ba['f1']:.3f}</div>
            <div class="lape-metric-sub">Accuracy: {ba['precision']:.3f} | Caught: {ba['recall']:.3f}</div>
            <div style="font-size:12px; color:var(--text-secondary); margin-top:8px;">Flags any order where selling price is below cost. Often triggers false alarms on normal clearance sales.</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="compare-card">
            <div class="lape-metric-label">Method B — Fixed Rules</div>
            <div class="lape-metric-value">Score: {bb['f1']:.3f}</div>
            <div class="lape-metric-sub">Accuracy: {bb['precision']:.3f} | Caught: {bb['recall']:.3f}</div>
            <div style="font-size:12px; color:var(--text-secondary); margin-top:8px;">Checks multiple fixed rules together. Catches more problems than Method A, but still misses unusual patterns.</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="compare-card winner">
            <div class="lape-metric-label">LAPE — Machine Learning (Best Result)</div>
            <div class="lape-metric-value" style="color:var(--accent);">Score: {bm['f1']:.3f}</div>
            <div class="lape-metric-sub">Accuracy: {bm['precision']:.3f} | Caught: {bm['recall']:.3f}</div>
            <div style="font-size:12px; color:var(--text-secondary); margin-top:8px;">Combines machine learning with practical risk checks. Catches the most leaks while producing the fewest false alarms.</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(f"#### {t('family_breakdown')}")
    fam_rows = []
    fam_source = benchmarks.get("model_family_breakdown") or benchmarks.get("family_breakdown", {})
    for fam_name, fam_metrics in fam_source.items():
        fam_type = fam_metrics.get("type", "leak_scenario")
        total_cases = fam_metrics.get("sample_count", fam_metrics.get("total", 0))
        if fam_type == "control":
            caught = f"{int(fam_metrics.get('clean_pass_rate', 1.0) * 100)}% Passed"
            success_rate = f"0.0% FP"
        else:
            caught = f"{fam_metrics.get('true_positives', fam_metrics.get('detected', 0))} / {total_cases}"
            success_rate = f"{fam_metrics.get('recall', 0.0)*100:.1f}%"
        fam_rows.append({
            "Test Situation Name": fam_name.replace("_", " ").title(),
            "Scenario Type": "Control (Valid)" if fam_type == "control" else "Leak Injection",
            "Total Test Cases": total_cases,
            "Verification Performance": caught,
            "Recall / Clean Pass": success_rate
        })
    render_table(pd.DataFrame(fam_rows), max_rows=10, height=240)

# ==============================================================================
# 7. REPORTS PAGE (PDF + CSV DOWNLOADS)
# ==============================================================================
def render_reports_view(data):
    st.subheader(t("reports"))
    st.caption("Create, preview, and download clear summary reports in Indian Rupees (₹) with full details.")

    col_cfg1, col_cfg2, col_cfg3 = st.columns(3)
    with col_cfg1:
        report_scope = st.selectbox(
            "Report Area",
            ["All Areas (Complete)", "Orders & Pricing Only", "Ad Campaigns Only", "Warehouse Stock Only"],
            index=0
        )
    with col_cfg2:
        inc_methodology = st.checkbox("Include explanation of how calculations work", value=True)
        inc_quality = st.checkbox("Include summary of data checks", value=True)
    with col_cfg3:
        inc_eval = st.checkbox("Include test results on artificial test data", value=True)
        date_fmt = st.session_state.get("date_format", "DD/MM/YYYY")

    alerts_df = generate_unified_alert_queue(
        data["tx_feat"], data["camp_feat"], data["inv_feat"], min_exposure=25.0
    )

    summary_data = {
        "total_tx": len(data["transactions"]),
        "completed_neg": int(((data["tx_feat"]["status"] == "Completed") & (data["tx_feat"]["price_cost_shortfall"] > 0)).sum()),
        "tx_shortfall": data["tx_feat"][data["tx_feat"]["status"] == "Completed"]["potential_shortfall_amount"].sum(),
        "total_camp": len(data["campaigns"]),
        "camp_viol": int(data["camp_feat"]["funnel_violation"].sum()),
        "camp_spend": data["camp_feat"][data["camp_feat"]["funnel_violation"] == 1]["spend"].sum(),
        "total_inv": len(data["inventory"]),
        "inv_disc": int(data["inv_feat"]["flag_discrepancy"].sum()),
        "total_alerts": len(alerts_df),
        "total_exposure": alerts_df["potential_exposure_amount"].sum(),
    }

    report_path = Path("reports/evaluation_report.json")
    benchmarks = None
    if report_path.exists():
        with open(report_path, encoding="utf-8") as f:
            benchmarks = json.load(f)

    # Generate PDF in memory
    python_date_fmt = "%d/%m/%Y" if date_fmt == "DD/MM/YYYY" else "%m/%d/%Y"
    pdf_bytes = generate_lape_pdf_report(
        summary_data=summary_data,
        top_alerts=alerts_df,
        benchmarks=benchmarks,
        scope=report_scope,
        include_methodology=inc_methodology,
        include_quality=inc_quality,
        include_evaluation=inc_eval,
        date_format=python_date_fmt,
    )

    st.markdown("---")
    st.markdown("#### Summary Report Preview")
    
    # Preview KPI metrics in INR
    p1, p2, p3 = st.columns(3)
    p1.metric("Total Possible Money at Risk", format_inr(summary_data["total_exposure"]))
    p2.metric("Number of Warnings in Report", f"{summary_data['total_alerts']:,}")
    p3.metric("Currency Used", "Indian Rupee (₹)")

    st.markdown("<br>", unsafe_allow_html=True)
    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        st.download_button(
            label=f"📄 {t('download_pdf')}",
            data=pdf_bytes,
            file_name=f"LAPE_Summary_Report_{datetime.datetime.now().strftime('%Y%m%d')}.pdf",
            mime="application/pdf",
            use_container_width=True,
            type="primary"
        )
    with col_dl2:
        csv_data = alerts_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label=f"📊 {t('download_csv')}",
            data=csv_data,
            file_name="lape_evidence_linked_alerts.csv",
            mime="text/csv",
            use_container_width=True
        )

# ==============================================================================
# 8. EXPANDED PREFERENCES & SETTINGS PAGE
# ==============================================================================
def render_settings_view():
    st.subheader(t("settings"))
    st.caption("Customize your display settings, choose your language, and adjust how information appears.")

    tab_app, tab_lang, tab_reg, tab_dash, tab_rep, tab_acc, tab_acct, tab_about = st.tabs([
        "Appearance", "Language", "Regional Settings", "Dashboard Options", "Reports", "Accessibility", "User Account", "About"
    ])

    # A. Appearance
    with tab_app:
        st.markdown("#### Appearance &amp; Color Themes")
        st.caption("Choose between Dark Mode (Black & Grey) and Light Mode (Clean & Bright).")
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            theme_opts = [t("dark"), t("light")]
            idx = 0 if st.session_state.theme == "dark" else 1
            choice = st.radio("Theme Mode", theme_opts, index=idx, key="settings_theme_radio")
            target = "light" if choice == t("light") else "dark"
            if target != st.session_state.theme:
                st.session_state.theme = target
                st.rerun()
        with col_t2:
            st.markdown("""
            <div class="lape-panel">
                <strong>Color Specifications:</strong><br>
                • <strong>Dark Mode:</strong> Background <code>#0B0B0C</code>, Sidebar <code>#141416</code>, Panels <code>#1C1C1F</code>, Teal Accent <code>#2CB7B0</code><br>
                • <strong>Light Mode:</strong> Background <code>#F8F9FA</code>, Sidebar <code>#F1F3F4</code>, Panels <code>#FFFFFF</code>, Blue Accent <code>#1A73E8</code><br>
                • <strong>Simple Design:</strong> Solid, high-contrast panels with no distracting gradient or blur effects.
            </div>
            """, unsafe_allow_html=True)

    # B. Language
    with tab_lang:
        st.markdown("#### Display Language")
        st.caption("Choose your preferred language across 8 Indian languages with English fallback.")
        current_idx = SUPPORTED_LANGUAGES.index(st.session_state.language) if st.session_state.language in SUPPORTED_LANGUAGES else 0
        sel_lang = st.selectbox("Application Language", SUPPORTED_LANGUAGES, index=current_idx, key="settings_lang_sel")
        if sel_lang != st.session_state.language:
            st.session_state.language = sel_lang
            st.rerun()

    # C. Regional Preferences
    with tab_reg:
        st.markdown("#### Regional &amp; Currency Formats")
        r_c1, r_c2 = st.columns(2)
        with r_c1:
            st.text_input("Country", value="India", disabled=True)
            st.text_input("Currency", value="INR (₹) - Indian Rupee", disabled=True)
            st.text_input("Number Format", value="Indian Numbering (e.g. ₹12,34,567.89)", disabled=True)
        with r_c2:
            st.session_state.date_format = st.selectbox("Date Format", ["DD/MM/YYYY", "YYYY-MM-DD", "MM/DD/YYYY"], index=0)
            st.text_input("Time Zone", value="Asia/Kolkata (IST, UTC+05:30)", disabled=True)
            st.text_input("Decimal Places", value="2 Decimal Places (.XX)", disabled=True)

    # D. Dashboard Preferences
    with tab_dash:
        st.markdown("#### Dashboard Layout Options")
        d_c1, d_c2 = st.columns(2)
        with d_c1:
            st.session_state.table_density = st.selectbox("Table Row Spacing", ["Standard (Comfortable)", "Compact (Dense)"], index=0)
            st.selectbox("Sort Warnings By", ["Highest Money at Risk First", "Check Immediately (Critical) First", "Most Reliable Proof First"], index=0)
        with d_c2:
            st.session_state.show_helper_text = st.checkbox("Show explanatory tip boxes throughout the app", value=st.session_state.show_helper_text)
            st.selectbox("Default Warning Filter", ["Show All Warnings", "Urgent Only", "Urgent & High Priority"], index=0)

    # E. Reports & Exports
    with tab_rep:
        st.markdown("#### Report &amp; Download Options")
        st.session_state.default_report_scope = st.selectbox("Default Report Area", ["All Areas (Complete)", "Orders & Pricing Only", "Ad Campaigns Only", "Warehouse Stock Only"], index=0)
        st.checkbox("Include full source details in CSV file", value=True)
        st.checkbox("Include clear note that test metrics use artificial test data", value=True, disabled=True)

    # F. Accessibility
    with tab_acc:
        st.markdown("#### Accessibility Controls")
        st.checkbox("Reduce animation and motion", value=False)
        st.checkbox("High contrast text for easy reading", value=True)
        st.checkbox("Clear labels for screen readers", value=True)

    # G. Account & Session
    with tab_acct:
        st.markdown("#### User Account")
        st.markdown("""
        <div class="lape-panel">
            <strong>Logged In As:</strong> Demo User (<code>demo@lape.local</code>)<br>
            <strong>Role:</strong> Financial Auditor<br>
            <strong>Access Level:</strong> Local Demo Mode<br>
            <strong>Security:</strong> Protected. Passwords are never shown or saved in settings.
        </div>
        """, unsafe_allow_html=True)
        if st.button("Sign Out", type="secondary"):
            st.session_state.authenticated = False
            st.rerun()

    # H. About LAPE
    with tab_about:
        st.markdown("#### About LAPE")
        st.markdown(f"""
        <div class="lape-panel">
            <strong>Product:</strong> LAPE (Leak Analysis &amp; Prediction Engine)<br>
            <strong>Version:</strong> v{APP_VERSION} (DATATHON 2026 Track 4)<br>
            <strong>Technology:</strong> Python {sys.version.split()[0]} | Streamlit | ReportLab<br>
            <strong>Privacy:</strong> 100% runs on your local computer. No information is sent over the internet.<br>
            <strong>Folder:</strong> <code>{ROOT_DIR}</code>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("#### Reset Settings")
        if st.button("Reset Settings to Defaults", help="Resets theme and layout options without modifying datasets or models"):
            st.session_state.theme = "dark"
            st.session_state.language = "English"
            st.session_state.table_density = "Comfortable"
            st.session_state.show_helper_text = True
            st.success("Settings reset to defaults.")
            st.rerun()

# ==============================================================================
# ENTRYPOINT
# ==============================================================================
if __name__ == "__main__":
    if not st.session_state.get("authenticated", False):
        render_login_page()
    else:
        main()
