# ============================================================================
# LEAF GUARD AI - Professional Agricultural Intelligence Platform
# Advanced Plant Disease Detection & Farm Management System
# Languages: English, Hindi, Bengali
# Accuracy: 94.37% | Classes: 10+ | Model: MobileNetV2
# ============================================================================

import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import os
from datetime import datetime
import base64
from io import BytesIO

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Leaf Guard AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': "Leaf Guard AI - Advanced Agricultural Intelligence Platform"
    }
)


# ============================================================================
# COMPREHENSIVE TRANSLATIONS
# ============================================================================

TRANSLATIONS = {
    'english': {
        'title': 'LEAF GUARD AI',
        'subtitle': 'Advanced Agricultural Intelligence Platform',
        'tagline': 'Empowering Farmers with AI-Driven Precision Agriculture',
        'scan': 'Scan & Detect',
        'cost_calc': 'Cost Calculator',
        'crop_calendar': 'Crop Calendar',
        'history': 'History',
        'detectable_diseases': 'Detectable Diseases',
        'upload_image': 'Upload Plant Image',
        'drag_drop': 'Drag and drop or click to upload',
        'upload_help': 'Upload a clear image of the affected leaf for instant AI analysis',
        'uploaded_image': 'Uploaded Image',
        'ai_analysis': 'AI Analysis',
        'analyze_disease': 'Analyze Disease',
        'analyzing': 'AI is analyzing your image...',
        'confidence_level': 'Confidence Level',
        'detection_probabilities': 'Detection Probabilities',
        'disease_symptoms': 'Disease Symptoms',
        'organic_treatment': 'Organic Treatment',
        'chemical_treatment': 'Chemical Treatment',
        'prevention_strategies': 'Prevention Strategies',
        'key_facts': 'Key Facts',
        'analysis_complete': 'Analysis complete! Results saved to history.',
        'upload_to_begin': 'Upload a leaf image to begin AI analysis',
        'upload_guidelines': 'Upload Guidelines',
        'guideline_1': 'Use well-lit, clear images',
        'guideline_2': 'Focus on the affected leaf area',
        'guideline_3': 'Capture the entire leaf when possible',
        'guideline_4': 'Avoid blurry or extremely dark images',
        'guideline_5': 'One leaf per image for best accuracy',
        'disease_database': 'Comprehensive Disease Database',
        'database_desc': 'Our advanced AI model can accurately identify 10+ plant diseases across major agricultural crops. Simply upload a clear photograph of the affected leaf for instant diagnosis and treatment recommendations.',
        'best_practices': 'Best Practices for Accurate Detection',
        'lighting': 'Lighting',
        'lighting_desc': 'Take photos in bright, natural daylight',
        'focus': 'Focus',
        'focus_desc': 'Ensure the affected area is clearly visible',
        'distance': 'Distance',
        'distance_desc': 'Capture from 6-12 inches away',
        'angle': 'Angle',
        'angle_desc': 'Photograph straight on, not at an angle',
        'background': 'Background',
        'background_desc': 'Use plain background if possible',
        'cost_calculator': 'Treatment Cost Calculator',
        'input_parameters': 'Input Parameters',
        'select_disease': 'Select Disease for Treatment Cost Estimation',
        'affected_area': 'Affected Area (acres)',
        'area_help': 'Enter the total area requiring treatment',
        'treatment_preference': 'Treatment Preference',
        'treatment_help': 'Select your preferred treatment approach',
        'organic': 'Organic',
        'chemical': 'Chemical',
        'integrated': 'Integrated (Both)',
        'calculate_cost': 'Calculate Treatment Cost',
        'cost_analysis': 'Cost Analysis for',
        'acres': 'acre(s)',
        'total_cost': 'Total Estimated Cost',
        'cost_breakdown': 'Detailed Cost Breakdown',
        'materials': 'Materials',
        'labor': 'Labor',
        'equipment': 'Equipment',
        'treatment_protocol': 'Treatment Protocol',
        'selected_treatment': 'Selected Treatment:',
        'recommended_products': 'Recommended Products:',
        'cost_note': 'Note: Costs are estimates based on current market rates. Actual costs may vary by location, disease severity, and product availability.',
        'cost_components': 'Cost Components',
        'materials_list': 'Fungicides/Bactericides',
        'copper_compounds': 'Copper compounds',
        'organic_treatments': 'Organic treatments',
        'protective_equipment': 'Protective equipment',
        'application_time': 'Application time',
        'plant_removal': 'Infected plant removal',
        'field_monitoring': 'Field monitoring',
        'post_treatment': 'Post-treatment care',
        'sprayers': 'Sprayers & applicators',
        'hand_tools': 'Hand tools',
        'safety_gear': 'Safety gear',
        'storage': 'Storage containers',
        'savings_tips': 'Savings Tips',
        'tip_1': 'Implement preventive measures early',
        'tip_2': 'Join farmer cooperatives for bulk discounts',
        'tip_3': 'Early detection reduces treatment costs',
        'tip_4': 'Practice crop rotation to minimize disease',
        'crop_planning': 'Comprehensive Crop Planning Guide',
        'select_crop': 'Select Crop for Detailed Information',
        'growing_guide': 'Complete Growing Guide',
        'planting_season': 'Planting Season',
        'harvest_season': 'Harvest Season',
        'optimal_temp': 'Optimal Temperature',
        'water_requirements': 'Water Requirements',
        'plant_spacing': 'Plant Spacing',
        'soil_ph': 'Soil pH',
        'days_to_harvest': 'Days to Harvest',
        'best_season': 'Best Season',
        'monthly_action': 'Monthly Action Plan',
        'planting': 'Planting:',
        'planting_desc': 'Review calendar above for suitable crops this month',
        'maintenance': 'Maintenance:',
        'maintenance_desc': 'Regular monitoring for pests and diseases',
        'soil_prep': 'Soil Preparation:',
        'soil_prep_desc': 'Prepare beds for upcoming planting',
        'fertilization': 'Fertilization:',
        'fertilization_desc': 'Follow growth stage-specific schedule',
        'scan_history': 'Scan History & Performance Analytics',
        'no_history': 'No Scan History Available',
        'no_history_desc': 'Start by scanning a leaf in the Scan & Detect tab to build your analysis history.',
        'total_scans': 'Total Scans',
        'avg_confidence': 'Average Confidence',
        'clear_history': 'Clear All History',
        'history_cleared': 'History cleared successfully!',
        'disease_stats': 'Disease Detection Statistics',
        'scan_records': 'Detailed Scan Records',
        'confidence': 'Confidence:',
        'showing_latest': 'Showing latest 20 of',
        'total_scans_text': 'total scans.',
        'model_accuracy': 'Model Accuracy',
        'detectable': 'Detectable Diseases',
        'tomato': 'Tomato',
        'potato': 'Potato',
        'pepper': 'Pepper',
        'corn': 'Corn',
        'severity_high': 'High',
        'severity_medium': 'Medium',
        'severity_low': 'Low',
    },
    'hindi': {
        'title': 'लीफ गार्ड AI',
        'subtitle': 'उन्नत कृषि बुद्धिमत्ता मंच',
        'tagline': 'AI-संचालित सटीक कृषि के साथ किसानों को सशक्त बनाना',
        'scan': 'स्कैन और पहचान',
        'cost_calc': 'लागत कैलकुलेटर',
        'crop_calendar': 'फसल कैलेंडर',
        'history': 'इतिहास',
        'detectable_diseases': 'पहचानने योग्य रोग',
        'upload_image': 'पौधे की छवि अपलोड करें',
        'drag_drop': 'खींचें और छोड़ें या अपलोड करने के लिए क्लिक करें',
        'upload_help': 'तत्काल AI विश्लेषण के लिए प्रभावित पत्ती की स्पष्ट छवि अपलोड करें',
        'uploaded_image': 'अपलोड की गई छवि',
        'ai_analysis': 'AI विश्लेषण',
        'analyze_disease': 'रोग का विश्लेषण करें',
        'analyzing': 'AI आपकी छवि का विश्लेषण कर रहा है...',
        'confidence_level': 'विश्वास स्तर',
        'detection_probabilities': 'पहचान संभावनाएं',
        'disease_symptoms': 'रोग के लक्षण',
        'organic_treatment': 'जैविक उपचार',
        'chemical_treatment': 'रासायनिक उपचार',
        'prevention_strategies': 'रोकथाम रणनीतियां',
        'key_facts': 'मुख्य तथ्य',
        'analysis_complete': 'विश्लेषण पूर्ण! परिणाम इतिहास में सहेजे गए।',
        'upload_to_begin': 'AI विश्लेषण शुरू करने के लिए पत्ती की छवि अपलोड करें',
        'upload_guidelines': 'अपलोड दिशानिर्देश',
        'guideline_1': 'अच्छी रोशनी वाली स्पष्ट छवियों का उपयोग करें',
        'guideline_2': 'प्रभावित पत्ती क्षेत्र पर ध्यान केंद्रित करें',
        'guideline_3': 'जब संभव हो तो पूरी पत्ती को कैप्चर करें',
        'guideline_4': 'धुंधली या बहुत अंधेरी छवियों से बचें',
        'guideline_5': 'सर्वोत्तम सटीकता के लिए प्रति छवि एक पत्ती',
        'disease_database': 'व्यापक रोग डेटाबेस',
        'database_desc': 'हमारा उन्नत AI मॉडल प्रमुख कृषि फसलों में 10+ पौधों की बीमारियों को सटीक रूप से पहचान सकता है। तत्काल निदान और उपचार सिफारिशों के लिए बस प्रभावित पत्ती की एक स्पष्ट तस्वीर अपलोड करें।',
        'best_practices': 'सटीक पहचान के लिए सर्वोत्तम प्रथाएं',
        'lighting': 'प्रकाश',
        'lighting_desc': 'उज्ज्वल, प्राकृतिक दिन के उजाले में फोटो लें',
        'focus': 'फोकस',
        'focus_desc': 'सुनिश्चित करें कि प्रभावित क्षेत्र स्पष्ट रूप से दिखाई दे',
        'distance': 'दूरी',
        'distance_desc': '6-12 इंच दूर से कैप्चर करें',
        'angle': 'कोण',
        'angle_desc': 'सीधे फोटो लें, कोण पर नहीं',
        'background': 'पृष्ठभूमि',
        'background_desc': 'यदि संभव हो तो सादे पृष्ठभूमि का उपयोग करें',
        'cost_calculator': 'उपचार लागत कैलकुलेटर',
        'input_parameters': 'इनपुट पैरामीटर',
        'select_disease': 'उपचार लागत अनुमान के लिए रोग चुनें',
        'affected_area': 'प्रभावित क्षेत्र (एकड़)',
        'area_help': 'उपचार की आवश्यकता वाले कुल क्षेत्र दर्ज करें',
        'treatment_preference': 'उपचार वरीयता',
        'treatment_help': 'अपनी पसंदीदा उपचार विधि चुनें',
        'organic': 'जैविक',
        'chemical': 'रासायनिक',
        'integrated': 'एकीकृत (दोनों)',
        'calculate_cost': 'उपचार लागत की गणना करें',
        'cost_analysis': 'लागत विश्लेषण',
        'acres': 'एकड़',
        'total_cost': 'कुल अनुमानित लागत',
        'cost_breakdown': 'विस्तृत लागत विवरण',
        'materials': 'सामग्री',
        'labor': 'श्रम',
        'equipment': 'उपकरण',
        'treatment_protocol': 'उपचार प्रोटोकॉल',
        'selected_treatment': 'चयनित उपचार:',
        'recommended_products': 'अनुशंसित उत्पाद:',
        'cost_note': 'नोट: लागत वर्तमान बाजार दरों के आधार पर अनुमान हैं। वास्तविक लागत स्थान, रोग की गंभीरता और उत्पाद उपलब्धता के अनुसार भिन्न हो सकती है।',
        'cost_components': 'लागत घटक',
        'materials_list': 'फफूंदनाशक/जीवाणुनाशक',
        'copper_compounds': 'तांबा यौगिक',
        'organic_treatments': 'जैविक उपचार',
        'protective_equipment': 'सुरक्षात्मक उपकरण',
        'application_time': 'आवेदन का समय',
        'plant_removal': 'संक्रमित पौधे को हटाना',
        'field_monitoring': 'क्षेत्र निगरानी',
        'post_treatment': 'उपचार के बाद देखभाल',
        'sprayers': 'स्प्रेयर और एप्लीकेटर',
        'hand_tools': 'हाथ के औजार',
        'safety_gear': 'सुरक्षा गियर',
        'storage': 'भंडारण कंटेनर',
        'savings_tips': 'बचत युक्तियाँ',
        'tip_1': 'जल्दी निवारक उपाय लागू करें',
        'tip_2': 'थोक छूट के लिए किसान सहकारी समितियों में शामिल हों',
        'tip_3': 'प्रारंभिक पहचान उपचार लागत कम करती है',
        'tip_4': 'रोग को कम करने के लिए फसल चक्र का अभ्यास करें',
        'crop_planning': 'व्यापक फसल योजना गाइड',
        'select_crop': 'विस्तृत जानकारी के लिए फसल चुनें',
        'growing_guide': 'संपूर्ण उगाने की गाइड',
        'planting_season': 'रोपण का मौसम',
        'harvest_season': 'कटाई का मौसम',
        'optimal_temp': 'इष्टतम तापमान',
        'water_requirements': 'पानी की आवश्यकताएं',
        'plant_spacing': 'पौधे की दूरी',
        'soil_ph': 'मिट्टी pH',
        'days_to_harvest': 'कटाई के दिन',
        'best_season': 'सर्वोत्तम मौसम',
        'monthly_action': 'मासिक कार्य योजना',
        'planting': 'रोपण:',
        'planting_desc': 'इस महीने उपयुक्त फसलों के लिए ऊपर कैलेंडर देखें',
        'maintenance': 'रखरखाव:',
        'maintenance_desc': 'कीटों और रोगों के लिए नियमित निगरानी',
        'soil_prep': 'मिट्टी की तैयारी:',
        'soil_prep_desc': 'आगामी रोपण के लिए बेड तैयार करें',
        'fertilization': 'उर्वरक:',
        'fertilization_desc': 'वृद्धि चरण-विशिष्ट अनुसूची का पालन करें',
        'scan_history': 'स्कैन इतिहास और प्रदर्शन विश्लेषण',
        'no_history': 'कोई स्कैन इतिहास उपलब्ध नहीं',
        'no_history_desc': 'अपना विश्लेषण इतिहास बनाने के लिए स्कैन और पहचान टैब में एक पत्ती को स्कैन करके शुरू करें।',
        'total_scans': 'कुल स्कैन',
        'avg_confidence': 'औसत विश्वास',
        'clear_history': 'सभी इतिहास साफ़ करें',
        'history_cleared': 'इतिहास सफलतापूर्वक साफ़ हो गया!',
        'disease_stats': 'रोग पहचान सांख्यिकी',
        'scan_records': 'विस्तृत स्कैन रिकॉर्ड',
        'confidence': 'विश्वास:',
        'showing_latest': 'नवीनतम 20 दिखा रहे हैं',
        'total_scans_text': 'कुल स्कैन।',
        'model_accuracy': 'मॉडल सटीकता',
        'detectable': 'पहचानने योग्य रोग',
        'tomato': 'टमाटर',
        'potato': 'आलू',
        'pepper': 'मिर्च',
        'corn': 'मक्का',
        'severity_high': 'उच्च',
        'severity_medium': 'मध्यम',
        'severity_low': 'निम्न',
    },
    'bengali': {
        'title': 'লীফ গার্ড AI',
        'subtitle': 'উন্নত কৃষি বুদ্ধিমত্তা প্ল্যাটফর্ম',
        'tagline': 'AI-চালিত নির্ভুল কৃষির মাধ্যমে কৃষকদের ক্ষমতায়ন',
        'scan': 'স্ক্যান এবং সনাক্ত করুন',
        'cost_calc': 'খরচ ক্যালকুলেটর',
        'crop_calendar': 'ফসল ক্যালেন্ডার',
        'history': 'ইতিহাস',
        'detectable_diseases': 'সনাক্তযোগ্য রোগ',
        'upload_image': 'উদ্ভিদের ছবি আপলোড করুন',
        'drag_drop': 'টেনে আনুন এবং ড্রপ করুন বা আপলোড করতে ক্লিক করুন',
        'upload_help': 'তাত্ক্ষণিক AI বিশ্লেষণের জন্য প্রভাবিত পাতার একটি স্পষ্ট ছবি আপলোড করুন',
        'uploaded_image': 'আপলোড করা ছবি',
        'ai_analysis': 'AI বিশ্লেষণ',
        'analyze_disease': 'রোগ বিশ্লেষণ করুন',
        'analyzing': 'AI আপনার ছবি বিশ্লেষণ করছে...',
        'confidence_level': 'আত্মবিশ্বাসের স্তর',
        'detection_probabilities': 'সনাক্তকরণ সম্ভাবনা',
        'disease_symptoms': 'রোগের লক্ষণ',
        'organic_treatment': 'জৈব চিকিৎসা',
        'chemical_treatment': 'রাসায়নিক চিকিৎসা',
        'prevention_strategies': 'প্রতিরোধ কৌশল',
        'key_facts': 'মূল তথ্য',
        'analysis_complete': 'বিশ্লেষণ সম্পূর্ণ! ফলাফল ইতিহাসে সংরক্ষিত।',
        'upload_to_begin': 'AI বিশ্লেষণ শুরু করতে একটি পাতার ছবি আপলোড করুন',
        'upload_guidelines': 'আপলোড নির্দেশিকা',
        'guideline_1': 'ভাল আলোযুক্ত, পরিষ্কার ছবি ব্যবহার করুন',
        'guideline_2': 'প্রভাবিত পাতার এলাকায় ফোকাস করুন',
        'guideline_3': 'সম্ভব হলে সম্পূর্ণ পাতা ক্যাপচার করুন',
        'guideline_4': 'ঝাপসা বা অত্যন্ত অন্ধকার ছবি এড়িয়ে চলুন',
        'guideline_5': 'সর্বোত্তম নির্ভুলতার জন্য প্রতি ছবিতে একটি পাতা',
        'disease_database': 'বিস্তৃত রোগ ডেটাবেস',
        'database_desc': 'আমাদের উন্নত AI মডেল প্রধান কৃষি ফসলের মধ্যে 10+ উদ্ভিদ রোগ সঠিকভাবে সনাক্ত করতে পারে। তাত্ক্ষণিক নির্ণয় এবং চিকিৎসা সুপারিশের জন্য শুধুমাত্র প্রভাবিত পাতার একটি স্পষ্ট ছবি আপলোড করুন।',
        'best_practices': 'সঠিক সনাক্তকরণের জন্য সেরা অনুশীলন',
        'lighting': 'আলো',
        'lighting_desc': 'উজ্জ্বল, প্রাকৃতিক দিনের আলোতে ফটো তুলুন',
        'focus': 'ফোকাস',
        'focus_desc': 'নিশ্চিত করুন যে প্রভাবিত এলাকা স্পষ্টভাবে দৃশ্যমান',
        'distance': 'দূরত্ব',
        'distance_desc': '6-12 ইঞ্চি দূর থেকে ক্যাপচার করুন',
        'angle': 'কোণ',
        'angle_desc': 'সরাসরি ফটোগ্রাফ করুন, কোণে নয়',
        'background': 'পটভূমি',
        'background_desc': 'সম্ভব হলে সাধারণ পটভূমি ব্যবহার করুন',
        'cost_calculator': 'চিকিৎসা খরচ ক্যালকুলেটর',
        'input_parameters': 'ইনপুট প্যারামিটার',
        'select_disease': 'চিকিৎসা খরচ অনুমানের জন্য রোগ নির্বাচন করুন',
        'affected_area': 'প্রভাবিত এলাকা (একর)',
        'area_help': 'চিকিৎসার প্রয়োজন মোট এলাকা লিখুন',
        'treatment_preference': 'চিকিৎসা পছন্দ',
        'treatment_help': 'আপনার পছন্দের চিকিৎসা পদ্ধতি নির্বাচন করুন',
        'organic': 'জৈব',
        'chemical': 'রাসায়নিক',
        'integrated': 'সমন্বিত (উভয়)',
        'calculate_cost': 'চিকিৎসা খরচ গণনা করুন',
        'cost_analysis': 'খরচ বিশ্লেষণ',
        'acres': 'একর',
        'total_cost': 'মোট আনুমানিক খরচ',
        'cost_breakdown': 'বিস্তারিত খরচ বিবরণ',
        'materials': 'উপকরণ',
        'labor': 'শ্রম',
        'equipment': 'সরঞ্জাম',
        'treatment_protocol': 'চিকিৎসা প্রোটোকল',
        'selected_treatment': 'নির্বাচিত চিকিৎসা:',
        'recommended_products': 'প্রস্তাবিত পণ্য:',
        'cost_note': 'নোট: খরচ বর্তমান বাজার হারের উপর ভিত্তি করে অনুমান। প্রকৃত খরচ অবস্থান, রোগের তীব্রতা এবং পণ্য প্রাপ্যতার উপর নির্ভর করে পরিবর্তিত হতে পারে।',
        'cost_components': 'খরচ উপাদান',
        'materials_list': 'ছত্রাকনাশক/ব্যাকটেরিয়ানাশক',
        'copper_compounds': 'তামা যৌগ',
        'organic_treatments': 'জৈব চিকিৎসা',
        'protective_equipment': 'সুরক্ষামূলক সরঞ্জাম',
        'application_time': 'প্রয়োগের সময়',
        'plant_removal': 'সংক্রমিত উদ্ভিদ অপসারণ',
        'field_monitoring': 'মাঠ পর্যবেক্ষণ',
        'post_treatment': 'চিকিৎসা পরবর্তী যত্ন',
        'sprayers': 'স্প্রেয়ার এবং অ্যাপ্লিকেটর',
        'hand_tools': 'হাতের সরঞ্জাম',
        'safety_gear': 'নিরাপত্তা গিয়ার',
        'storage': 'সংরক্ষণ পাত্র',
        'savings_tips': 'সঞ্চয় টিপস',
        'tip_1': 'প্রাথমিক প্রতিরোধমূলক ব্যবস্থা প্রয়োগ করুন',
        'tip_2': 'থোক ছাড়ের জন্য কৃষক সমবায়ে যোগ দিন',
        'tip_3': 'প্রাথমিক সনাক্তকরণ চিকিৎসা খরচ হ্রাস করে',
        'tip_4': 'রোগ কমাতে ফসল ঘূর্ণন অনুশীলন করুন',
        'crop_planning': 'বিস্তৃত ফসল পরিকল্পনা গাইড',
        'select_crop': 'বিস্তারিত তথ্যের জন্য ফসল নির্বাচন করুন',
        'growing_guide': 'সম্পূর্ণ ক্রমবর্ধমান গাইড',
        'planting_season': 'রোপণ মৌসুম',
        'harvest_season': 'ফসল কাটার মৌসুম',
        'optimal_temp': 'সর্বোত্তম তাপমাত্রা',
        'water_requirements': 'জলের প্রয়োজনীয়তা',
        'plant_spacing': 'উদ্ভিদ দূরত্ব',
        'soil_ph': 'মাটি pH',
        'days_to_harvest': 'ফসল কাটার দিন',
        'best_season': 'সেরা মৌসুম',
        'monthly_action': 'মাসিক কর্ম পরিকল্পনা',
        'planting': 'রোপণ:',
        'planting_desc': 'এই মাসে উপযুক্ত ফসলের জন্য উপরে ক্যালেন্ডার পর্যালোচনা করুন',
        'maintenance': 'রক্ষণাবেক্ষণ:',
        'maintenance_desc': 'কীটপতঙ্গ এবং রোগের জন্য নিয়মিত পর্যবেক্ষণ',
        'soil_prep': 'মাটি প্রস্তুতি:',
        'soil_prep_desc': 'আগামী রোপণের জন্য বিছানা প্রস্তুত করুন',
        'fertilization': 'সার:',
        'fertilization_desc': 'বৃদ্ধির পর্যায়-নির্দিষ্ট সময়সূচী অনুসরণ করুন',
        'scan_history': 'স্ক্যান ইতিহাস এবং কর্মক্ষমতা বিশ্লেষণ',
        'no_history': 'কোন স্ক্যান ইতিহাস উপলব্ধ নেই',
        'no_history_desc': 'আপনার বিশ্লেষণ ইতিহাস তৈরি করতে স্ক্যান এবং সনাক্ত করুন ট্যাবে একটি পাতা স্ক্যান করে শুরু করুন।',
        'total_scans': 'মোট স্ক্যান',
        'avg_confidence': 'গড় আত্মবিশ্বাস',
        'clear_history': 'সমস্ত ইতিহাস সাফ করুন',
        'history_cleared': 'ইতিহাস সফলভাবে সাফ হয়েছে!',
        'disease_stats': 'রোগ সনাক্তকরণ পরিসংখ্যান',
        'scan_records': 'বিস্তারিত স্ক্যান রেকর্ড',
        'confidence': 'আত্মবিশ্বাস:',
        'showing_latest': 'সর্বশেষ 20 দেখাচ্ছে',
        'total_scans_text': 'মোট স্ক্যান।',
        'model_accuracy': 'মডেল নির্ভুলতা',
        'detectable': 'সনাক্তযোগ্য রোগ',
        'tomato': 'টমেটো',
        'potato': 'আলু',
        'pepper': 'মরিচ',
        'corn': 'ভুট্টা',
        'severity_high': 'উচ্চ',
        'severity_medium': 'মাঝারি',
        'severity_low': 'নিম্ন',
    }
}

# ============================================================================
# ENHANCED WARM COLOR SCHEME CSS WITH FIXED DROPDOWN VISIBILITY
# ============================================================================

def load_css():
    st.markdown("""
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=5.0, user-scalable=yes">
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
        
        * {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }
        
        /* 60% Primary - Navy Blue Background */
        .stApp {
            background: linear-gradient(135deg, #0a1628 0%, #1a2a42 50%, #243447 100%);
            background-attachment: fixed;
        }
        
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        .stDeployButton {display: none;}
        header {visibility: hidden;}
        
        /* Hero Section - Navy with White text and Gold accents */
        .hero-section {
            text-align: center;
            padding: 3rem 2rem;
            background: linear-gradient(135deg, rgba(255, 255, 255, 0.05) 0%, rgba(255, 255, 255, 0.02) 100%);
            border-radius: 20px;
            margin-bottom: 2.5rem;
            border: 1px solid rgba(212, 175, 55, 0.3);
            box-shadow: 0 15px 40px rgba(0, 0, 0, 0.5);
        }
        
        /* 10% Accent - Gold Gradient for Title */
        .hero-title {
            font-size: 3.5rem;
            font-weight: 800;
            background: linear-gradient(135deg, #FFD700 0%, #FFA500 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 0.75rem;
            letter-spacing: 0.02em;
        }
        
        /* 30% Secondary - White/Light text */
        .hero-subtitle {
            font-size: 1.2rem;
            color: rgba(255, 255, 255, 0.9);
            font-weight: 500;
            margin-bottom: 0.5rem;
        }
        
        .hero-tagline {
            font-size: 1rem;
            color: rgba(255, 255, 255, 0.7);
            font-weight: 400;
            max-width: 600px;
            margin: 0 auto;
        }
        
        .stats-banner {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1.5rem;
            margin: 2rem 0;
        }
        
        /* Cards - Navy with White borders and Gold accents */
        .stat-card {
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(212, 175, 55, 0.3);
            border-radius: 16px;
            padding: 1.75rem;
            text-align: center;
            transition: all 0.3s ease;
            backdrop-filter: blur(10px);
        }
        
        .stat-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 15px 30px rgba(212, 175, 55, 0.2);
            border-color: rgba(212, 175, 55, 0.6);
            background: rgba(255, 255, 255, 0.05);
        }
        
        .stat-value {
            font-size: 2.5rem;
            font-weight: 800;
            background: linear-gradient(135deg, #FFD700 0%, #FFA500 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 0.5rem;
        }
        
        .stat-label {
            font-size: 0.95rem;
            color: rgba(255, 255, 255, 0.75);
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }
        
        .feature-card {
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 16px;
            padding: 2rem;
            margin: 1.25rem 0;
            transition: all 0.3s ease;
            backdrop-filter: blur(10px);
        }
        
        .feature-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 12px 30px rgba(0, 0, 0, 0.3);
            border-color: rgba(212, 175, 55, 0.3);
            background: rgba(255, 255, 255, 0.04);
        }
        
        .result-card {
            background: linear-gradient(135deg, rgba(255, 255, 255, 0.05) 0%, rgba(255, 255, 255, 0.02) 100%);
            border: 1.5px solid rgba(212, 175, 55, 0.4);
            border-radius: 20px;
            padding: 2.5rem;
            margin: 1.5rem 0;
            box-shadow: 0 20px 50px rgba(0, 0, 0, 0.4);
            backdrop-filter: blur(10px);
        }
        
        /* 10% Accent - Gold for important badges */
        .disease-badge {
            display: inline-block;
            background: linear-gradient(135deg, #FFD700 0%, #FFA500 100%);
            color: #0a1628;
            padding: 0.85rem 2rem;
            border-radius: 40px;
            font-size: 1.4rem;
            font-weight: 700;
            margin: 1rem 0;
            box-shadow: 0 8px 20px rgba(255, 215, 0, 0.3);
            letter-spacing: 0.01em;
        }
        
        .confidence-meter {
            background: rgba(255, 255, 255, 0.03);
            border-radius: 16px;
            padding: 1.75rem;
            margin: 1.25rem 0;
            text-align: center;
            border: 1px solid rgba(212, 175, 55, 0.2);
        }
        
        .confidence-value {
            font-size: 3.5rem;
            font-weight: 800;
            background: linear-gradient(135deg, #FFD700 0%, #FFA500 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            letter-spacing: -0.02em;
        }
        
        .info-section {
            background: rgba(255, 255, 255, 0.02);
            border-radius: 12px;
            padding: 1.25rem;
            margin: 0.85rem 0;
            border-left: 3px solid #D4AF37;
            transition: all 0.3s ease;
        }
        
        .info-section:hover {
            background: rgba(255, 255, 255, 0.04);
            border-left-color: #FFD700;
        }
        
        .section-title {
            font-size: 1.15rem;
            color: #FFD700;
            font-weight: 700;
            margin-bottom: 0.85rem;
        }
        
        .section-content {
            color: rgba(255, 255, 255, 0.85);
            font-size: 1rem;
            line-height: 1.7;
        }
        
        /* 10% Accent - Gold buttons (CTA) */
        .stButton > button {
            background: linear-gradient(135deg, #FFD700 0%, #FFA500 100%);
            color: #0a1628 !important;
            font-size: 1rem;
            font-weight: 700;
            padding: 0.8rem 2rem;
            border-radius: 10px;
            border: none;
            box-shadow: 0 6px 20px rgba(255, 215, 0, 0.3);
            transition: all 0.3s ease;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 30px rgba(255, 215, 0, 0.5);
            background: linear-gradient(135deg, #FFA500 0%, #FF8C00 100%);
        }
        
        /* Tabs - Navy background with Gold accents */
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.6rem;
            background: transparent;
            padding: 0.4rem;
        }
        
        .stTabs [data-baseweb="tab"] {
            background: rgba(255, 255, 255, 0.03);
            border-radius: 10px;
            color: rgba(255, 255, 255, 0.7) !important;
            font-size: 0.95rem;
            font-weight: 600;
            padding: 0.85rem 1.25rem;
            border: 1px solid rgba(255, 255, 255, 0.1);
            transition: all 0.3s ease;
        }
        
        .stTabs [data-baseweb="tab"]:hover {
            background: rgba(255, 255, 255, 0.05);
            color: rgba(255, 255, 255, 0.9) !important;
        }
        
        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, #FFD700 0%, #FFA500 100%);
            color: #0a1628 !important;
            border-color: transparent;
            box-shadow: 0 4px 12px rgba(255, 215, 0, 0.3);
        }
        
        .stProgress > div > div > div {
            background: linear-gradient(90deg, #FFD700 0%, #FFA500 100%);
            border-radius: 8px;
        }
        
        [data-testid="stFileUploader"] {
            background: rgba(255, 255, 255, 0.02);
            border: 2px dashed rgba(212, 175, 55, 0.4);
            border-radius: 16px;
            padding: 2.5rem;
            transition: all 0.3s ease;
        }
        
        [data-testid="stFileUploader"]:hover {
            border-color: rgba(212, 175, 55, 0.7);
            background: rgba(255, 255, 255, 0.04);
        }
        
        /* TEXT INPUT & NUMBER INPUT - White text on Navy background */
        .stTextInput > div > div > input,
        .stNumberInput > div > div > input {
            background: rgba(255, 255, 255, 0.05) !important;
            color: rgba(255, 255, 255, 0.95) !important;
            border: 1px solid rgba(212, 175, 55, 0.3) !important;
            border-radius: 10px !important;
            padding: 0.7rem !important;
            font-size: 0.95rem !important;
            transition: all 0.3s ease !important;
        }
        
        .stTextInput > div > div > input:focus,
        .stNumberInput > div > div > input:focus {
            border-color: #FFD700 !important;
            box-shadow: 0 0 0 2px rgba(255, 215, 0, 0.2) !important;
        }
        
        /* SELECTBOX - FULL VISIBILITY FIX */
        
        .stSelectbox {
            color: rgba(255, 255, 255, 0.95) !important;
        }
        
        .stSelectbox label {
            color: rgba(255, 255, 255, 0.9) !important;
        }
        
        .stSelectbox > div {
            background: transparent !important;
        }
        
        .stSelectbox > div > div {
            background: rgba(255, 255, 255, 0.05) !important;
            border: 1px solid rgba(212, 175, 55, 0.4) !important;
            border-radius: 10px !important;
        }
        
        .stSelectbox [data-baseweb="select"] {
            background: rgba(255, 255, 255, 0.05) !important;
        }
        
        .stSelectbox [data-baseweb="select"] > div {
            background: rgba(255, 255, 255, 0.05) !important;
            border-color: rgba(212, 175, 55, 0.4) !important;
            min-height: 42px !important;
        }
        
        .stSelectbox [data-baseweb="select"] > div > div:first-child {
            color: rgba(255, 255, 255, 0.95) !important;
        }
        
        .stSelectbox [data-baseweb="select"] span {
            color: rgba(255, 255, 255, 0.95) !important;
        }
        
        .stSelectbox [data-baseweb="select"] [data-baseweb="tag"] {
            background: transparent !important;
            color: rgba(255, 255, 255, 0.95) !important;
        }
        
        .stSelectbox svg {
            fill: rgba(255, 255, 255, 0.8) !important;
        }
        
        .stSelectbox [data-baseweb="popover"] {
            background: rgba(10, 22, 40, 0.98) !important;
        }
        
        .stSelectbox [role="listbox"] {
            background: rgba(10, 22, 40, 0.98) !important;
            border: 1px solid rgba(212, 175, 55, 0.4) !important;
            border-radius: 10px !important;
            backdrop-filter: blur(10px) !important;
        }
        
        .stSelectbox [role="option"] {
            background: transparent !important;
            color: rgba(255, 255, 255, 0.95) !important;
            padding: 0.75rem 1rem !important;
            transition: all 0.2s ease !important;
        }
        
        .stSelectbox [role="option"]:hover {
            background: rgba(212, 175, 55, 0.2) !important;
            color: #FFD700 !important;
        }
        
        .stSelectbox [aria-selected="true"] {
            background: rgba(212, 175, 55, 0.25) !important;
            color: #FFD700 !important;
            font-weight: 600 !important;
        }
        
        .stSelectbox input {
            color: rgba(255, 255, 255, 0.95) !important;
            background: transparent !important;
        }
        
        .disease-list {
            background: rgba(255, 255, 255, 0.02);
            border-radius: 10px;
            padding: 1rem;
            margin: 0.6rem 0;
            border-left: 3px solid #D4AF37;
            transition: all 0.3s ease;
        }
        
        .disease-list:hover {
            background: rgba(255, 255, 255, 0.04);
            transform: translateX(5px);
        }
        
        .crop-info-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 1.25rem;
            margin: 1.5rem 0;
        }
        
        .crop-info-item {
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(212, 175, 55, 0.2);
            border-radius: 12px;
            padding: 1.25rem;
            transition: all 0.3s ease;
        }
        
        .crop-info-item:hover {
            background: rgba(255, 255, 255, 0.05);
            border-color: rgba(212, 175, 55, 0.5);
            transform: translateY(-2px);
        }
        
        .crop-info-label {
            font-size: 0.85rem;
            color: #FFD700;
            font-weight: 600;
            margin-bottom: 0.5rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        
        .crop-info-value {
            font-size: 1.1rem;
            color: rgba(255, 255, 255, 0.9);
            font-weight: 500;
        }
        
        h1, h2, h3 {
            color: #FFD700 !important;
            font-weight: 700 !important;
            letter-spacing: -0.01em !important;
        }
        
        p, span, div, label {
            color: rgba(255, 255, 255, 0.85) !important;
        }
        
        img {
            border-radius: 12px;
            box-shadow: 0 6px 20px rgba(0, 0, 0, 0.4);
        }
        
        ::-webkit-scrollbar {
            width: 10px;
        }
        
        ::-webkit-scrollbar-track {
            background: rgba(255, 255, 255, 0.02);
        }
        
        ::-webkit-scrollbar-thumb {
            background: linear-gradient(135deg, #D4AF37 0%, #FFD700 100%);
            border-radius: 5px;
        }
        
        @keyframes fadeIn {
            from {
                opacity: 0;
                transform: translateY(15px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        .fade-in {
            animation: fadeIn 0.5s ease-out;
        }
        
        .severity-badge {
            display: inline-block;
            padding: 0.35rem 0.85rem;
            border-radius: 20px;
            font-weight: 600;
            font-size: 0.85rem;
            letter-spacing: 0.02em;
        }
        
        .severity-high {
            background: #ef4444;
            color: white;
        }
        
        .severity-medium {
            background: #f59e0b;
            color: white;
        }
        
        .severity-low {
            background: #10b981;
            color: white;
        }
        
        .stRadio > div {
            background: rgba(255, 255, 255, 0.02);
            padding: 1rem;
            border-radius: 10px;
        }
        
        .stRadio > div > label {
            color: rgba(255, 255, 255, 0.9) !important;
        }
        
        .stMetric {
            background: rgba(255, 255, 255, 0.02);
            padding: 1rem;
            border-radius: 10px;
        }
        
        .stMetric label {
            color: rgba(255, 255, 255, 0.7) !important;
        }
        
        .stMetric [data-testid="stMetricValue"] {
            color: #FFD700 !important;
        }
        
        .stAlert {
            background: rgba(255, 255, 255, 0.05);
            border-left: 4px solid #D4AF37;
            color: rgba(255, 255, 255, 0.9);
        }
        
                /* Tablets and below (768px) */
        @media screen and (max-width: 768px) {
            .hero-title {
                font-size: 2.5rem !important;
            }
            
            .hero-subtitle {
                font-size: 1rem !important;
            }
            
            .hero-tagline {
                font-size: 0.9rem !important;
            }
            
            .stats-banner {
                grid-template-columns: 1fr !important;
                gap: 1rem !important;
            }
            
            .stat-value {
                font-size: 2rem !important;
            }
            
            .stat-label {
                font-size: 0.85rem !important;
            }
            
            .disease-badge {
                font-size: 1.1rem !important;
                padding: 0.65rem 1.5rem !important;
            }
            
            .confidence-value {
                font-size: 2.5rem !important;
            }
            
            .crop-info-grid {
                grid-template-columns: 1fr !important;
            }
            
            .stTabs [data-baseweb="tab"] {
                font-size: 0.8rem !important;
                padding: 0.65rem 0.85rem !important;
            }
            
            .feature-card, .result-card {
                padding: 1.5rem !important;
            }
            
            .info-section {
                padding: 1rem !important;
            }
            
            h1 {
                font-size: 1.8rem !important;
            }
            
            h2 {
                font-size: 1.5rem !important;
            }
            
            h3 {
                font-size: 1.2rem !important;
            }
        }

        /* Mobile phones (480px) */
        @media screen and (max-width: 480px) {
            .hero-section {
                padding: 2rem 1rem !important;
            }
            
            .hero-title {
                font-size: 2rem !important;
                letter-spacing: 0 !important;
            }
            
            .hero-subtitle {
                font-size: 0.95rem !important;
            }
            
            .hero-tagline {
                font-size: 0.85rem !important;
            }
            
            .stat-card {
                padding: 1.25rem !important;
            }
            
            .stat-value {
                font-size: 1.8rem !important;
            }
            
            .stat-label {
                font-size: 0.75rem !important;
            }
            
            .disease-badge {
                font-size: 1rem !important;
                padding: 0.6rem 1.2rem !important;
            }
            
            .confidence-value {
                font-size: 2rem !important;
            }
            
            .confidence-meter {
                padding: 1.25rem !important;
            }
            
            .stButton > button {
                font-size: 0.9rem !important;
                padding: 0.7rem 1.5rem !important;
            }
            
            .stTabs [data-baseweb="tab"] {
                font-size: 0.75rem !important;
                padding: 0.6rem 0.7rem !important;
            }
            
            .feature-card, .result-card {
                padding: 1.2rem !important;
                margin: 1rem 0 !important;
            }
            
            .info-section {
                padding: 0.85rem !important;
            }
            
            .section-title {
                font-size: 1rem !important;
            }
            
            .section-content {
                font-size: 0.9rem !important;
            }
            
            .crop-info-item {
                padding: 1rem !important;
            }
            
            .crop-info-label {
                font-size: 0.75rem !important;
            }
            
            .crop-info-value {
                font-size: 1rem !important;
            }
            
            h1 {
                font-size: 1.5rem !important;
            }
            
            h2 {
                font-size: 1.3rem !important;
            }
            
            h3 {
                font-size: 1.1rem !important;
            }
            
            /* Make file uploader more mobile-friendly */
            [data-testid="stFileUploader"] {
                padding: 1.5rem 1rem !important;
            }
            
            /* Adjust selectbox for mobile */
            .stSelectbox [data-baseweb="select"] > div {
                min-height: 38px !important;
            }
            
            /* Better spacing for mobile forms */
            .stTextInput > div > div > input,
            .stNumberInput > div > div > input {
                font-size: 16px !important; /* Prevents zoom on iOS */
            }
        }

        /* Very small phones (360px) */
        @media screen and (max-width: 360px) {
            .hero-title {
                font-size: 1.75rem !important;
            }
            
            .hero-subtitle {
                font-size: 0.9rem !important;
            }
            
            .stat-value {
                font-size: 1.6rem !important;
            }
            
            .confidence-value {
                font-size: 1.75rem !important;
            }
            
            .disease-badge {
                font-size: 0.95rem !important;
                padding: 0.5rem 1rem !important;
            }
            
            .stTabs [data-baseweb="tab"] {
                font-size: 0.7rem !important;
                padding: 0.5rem 0.6rem !important;
            }
        }

        /* Landscape mode for mobile */
        @media screen and (max-height: 500px) and (orientation: landscape) {
            .hero-section {
                padding: 1.5rem 1rem !important;
            }
            
            .hero-title {
                font-size: 2rem !important;
            }
            
            .stats-banner {
                grid-template-columns: repeat(3, 1fr) !important;
            }
        }

        /* Touch-friendly adjustments */
        @media (hover: none) and (pointer: coarse) {
            .stButton > button {
                min-height: 44px !important; /* iOS touch target */
            }
            
            .stTabs [data-baseweb="tab"] {
                min-height: 44px !important;
            }
            
            .stSelectbox [data-baseweb="select"] > div {
                min-height: 44px !important;
            }
        }
        
    </style>
    """, unsafe_allow_html=True)

# ============================================================================
# SESSION STATE
# ============================================================================

if 'language' not in st.session_state:
    st.session_state.language = 'english'
if 'scan_history' not in st.session_state:
    st.session_state.scan_history = []

# ============================================================================
# DETECTABLE DISEASES DATABASE
# ============================================================================

DETECTABLE_DISEASES = {
    "Tomato": [
        {"name": "Late Blight", "severity": "High"},
        {"name": "Early Blight", "severity": "High"},
        {"name": "Septoria Leaf Spot", "severity": "Medium"},
        {"name": "Bacterial Spot", "severity": "High"},
        {"name": "Leaf Mold", "severity": "Medium"},
        {"name": "Mosaic Virus", "severity": "High"},
        {"name": "Target Spot", "severity": "Medium"},
        {"name": "Yellow Leaf Curl Virus", "severity": "High"},
    ],
    "Potato": [
        {"name": "Late Blight", "severity": "High"},
        {"name": "Early Blight", "severity": "High"},
    ],
    "Pepper": [
        {"name": "Bacterial Spot", "severity": "High"},
    ],
    "Corn": [
        {"name": "Common Rust", "severity": "Medium"},
    ]
}

# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

@st.cache_resource
def load_model():
    """Load the trained disease detection model"""
    try:
        return keras.models.load_model('leaf_guard_best.h5')
    except Exception as e:
        st.error(f"Model loading error: {str(e)}")
        return None

@st.cache_data
def load_class_names():
    """Load class names from file"""
    try:
        with open('class_names.txt', 'r') as f:
            return [line.strip() for line in f.readlines()]
    except FileNotFoundError:
        st.warning("Class names file not found. Using default classes.")
        return []

@st.cache_data
def load_disease_db():
    """Load comprehensive disease information database"""
    return {
        "late_blight": {
            "symptoms": "Water-soaked lesions, brown spots with white fungal growth on leaf undersides, rapid plant collapse.",
            "treatment": "Remove infected plants immediately. Apply copper-based fungicide. Improve air circulation between plants.",
            "prevention": "Use resistant varieties. Ensure good drainage. Maintain proper plant spacing (24-36 inches). Avoid overhead watering.",
            "facts": [
                "Caused the Irish Potato Famine (1845-1852)",
                "Can destroy entire crop within 7-10 days",
                "Produces 300,000+ spores per lesion daily",
                "Thrives in 90%+ humidity conditions"
            ],
            "cost_per_acre": {"materials": 150, "labor": 100, "equipment": 50},
            "organic_treatment": "Neem oil spray (2-3%), copper sulfate solution, baking soda mixture (1 tbsp per gallon water)",
            "chemical_treatment": "Mancozeb, Chlorothalonil, Metalaxyl-based fungicides"
        },
        "early_blight": {
            "symptoms": "Concentric ring patterns (target spots), yellowing leaves, lower leaves affected first, premature defoliation.",
            "treatment": "Remove affected leaves promptly. Apply fungicide every 7-10 days. Improve plant air circulation.",
            "prevention": "Practice 3-year crop rotation. Apply mulch around plants. Water at base only. Stake plants for airflow.",
            "facts": [
                "Most common tomato disease worldwide",
                "Creates distinctive bull's-eye pattern",
                "Progresses from lower to upper leaves",
                "Can cause 30-50% yield reduction"
            ],
            "cost_per_acre": {"materials": 120, "labor": 80, "equipment": 40},
            "organic_treatment": "Copper-based fungicides, neem oil applications, prompt removal of infected debris",
            "chemical_treatment": "Azoxystrobin, Chlorothalonil, Mancozeb"
        },
        "septoria_leaf_spot": {
            "symptoms": "Small circular spots (1/16 to 1/4 inch diameter), gray centers with dark borders, black pycnidia in centers.",
            "treatment": "Remove infected leaves immediately. Apply fungicide spray. Mulch soil surface. Water at plant base only.",
            "prevention": "Implement 3-4 year crop rotation. Ensure proper spacing. Remove all plant debris in fall. Avoid overhead irrigation.",
            "facts": [
                "Tomato-specific fungal pathogen",
                "Black dots are spore-producing structures",
                "Can survive on debris for multiple years",
                "Major problem in humid regions"
            ],
            "cost_per_acre": {"materials": 130, "labor": 90, "equipment": 45},
            "organic_treatment": "Copper fungicides, neem oil, thorough debris removal",
            "chemical_treatment": "Chlorothalonil, Mancozeb"
        },
        "bacterial_spot": {
            "symptoms": "Small dark spots with yellow halos, spots on leaves and fruit, leaf drop, reduced fruit quality.",
            "treatment": "Apply copper-based bactericide. Remove infected plant tissue. Improve air circulation between plants.",
            "prevention": "Use certified disease-free seeds. Practice 3-year rotation. Avoid overhead irrigation. Disinfect tools between uses.",
            "facts": [
                "Caused by 4 Xanthomonas species",
                "Spreads rapidly via water splash",
                "Can survive 1 year on plant debris",
                "Severely affects fruit marketability"
            ],
            "cost_per_acre": {"materials": 140, "labor": 95, "equipment": 48},
            "organic_treatment": "Copper sulfate solutions, immediate removal of infected plants",
            "chemical_treatment": "Copper-based bactericides, Actigard (SAR inducer)"
        },
        "common_rust": {
            "symptoms": "Small circular to elongate brown pustules on leaves, yellowing between pustules, premature leaf death.",
            "treatment": "Apply fungicide if severity exceeds 5% infection. Remove heavily infected leaves. Ensure good field airflow.",
            "prevention": "Plant resistant hybrid varieties. Apply balanced fertilization. Practice crop rotation. Avoid dense planting.",
            "facts": [
                "Caused by Puccinia sorghi fungus",
                "Pustules contain rust-colored spores",
                "Can reduce yields by 30-40%",
                "Spores are wind-dispersed"
            ],
            "cost_per_acre": {"materials": 110, "labor": 75, "equipment": 38},
            "organic_treatment": "Sulfur-based fungicides, resistant varieties",
            "chemical_treatment": "Azoxystrobin, Triazole-based fungicides"
        },
        "healthy": {
            "symptoms": "Vibrant green color, uniform growth pattern, no lesions or spots, strong stem structure, healthy root development.",
            "treatment": "Continue regular monitoring and good agricultural practices. No treatment needed.",
            "prevention": "Maintain optimal soil health. Provide proper nutrition (NPK 5-10-10). Ensure consistent watering schedule. Implement integrated pest management.",
            "facts": [
                "Prevention is more cost-effective than treatment",
                "Healthy plants have stronger disease immunity",
                "Regular field scouting prevents major outbreaks",
                "Soil health directly impacts plant health"
            ],
            "cost_per_acre": {"materials": 0, "labor": 0, "equipment": 0},
            "organic_treatment": "Compost tea applications, beneficial microorganism inoculation",
            "chemical_treatment": "None required - maintain preventive care"
        },
        "default": {
            "symptoms": "Unable to identify specific symptoms. Professional diagnosis recommended.",
            "treatment": "Obtain professional diagnosis before initiating treatment protocols.",
            "prevention": "Follow integrated pest management (IPM) best practices.",
            "facts": [
                "Early detection is crucial for effective management",
                "IPM reduces chemical dependency",
                "Consult agricultural extension services"
            ],
            "cost_per_acre": {"materials": 100, "labor": 80, "equipment": 40},
            "organic_treatment": "Consult agricultural expert",
            "chemical_treatment": "Consult agricultural expert"
        }
    }

@st.cache_data
def load_crop_calendar():
    """Load crop calendar data"""
    return {
        "tomato": {
            "planting": "March-May", 
            "harvest": "June-October",
            "temp": "20-30°C", 
            "water": "Moderate (1-2 inches/week)", 
            "season": "Spring-Summer",
            "spacing": "24-36 inches",
            "soil_ph": "6.0-6.8",
            "days_to_harvest": "60-85 days"
        },
        "potato": {
            "planting": "February-March", 
            "harvest": "May-July",
            "temp": "15-20°C", 
            "water": "Moderate-High (1-2 inches/week)", 
            "season": "Winter-Spring",
            "spacing": "12 inches",
            "soil_ph": "5.0-6.0",
            "days_to_harvest": "70-120 days"
        },
        "pepper": {
            "planting": "March-June", 
            "harvest": "July-October",
            "temp": "21-29°C", 
            "water": "Moderate (1-2 inches/week)", 
            "season": "Spring-Summer",
            "spacing": "18-24 inches",
            "soil_ph": "6.0-7.0",
            "days_to_harvest": "60-90 days"
        },
        "corn": {
            "planting": "April-June", 
            "harvest": "August-October",
            "temp": "16-30°C", 
            "water": "High (1.5 inches/week)", 
            "season": "Spring-Summer",
            "spacing": "8-12 inches",
            "soil_ph": "5.8-7.0",
            "days_to_harvest": "60-100 days"
        }
    }

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_disease_info(disease_name, disease_db):
    """Get detailed information for a disease"""
    for key in disease_db:
        if key in disease_name.lower():
            return disease_db[key]
    return disease_db['default']

def preprocess_image(image):
    """Preprocess image for model prediction"""
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

def predict_disease(model, image, class_names):
    """Predict disease from image and return top 3 predictions"""
    processed_img = preprocess_image(image)
    predictions = model.predict(processed_img, verbose=0)
    top_3_idx = np.argsort(predictions[0])[-3:][::-1]
    return [{
        'disease': class_names[idx],
        'confidence': float(predictions[0][idx]) * 100
    } for idx in top_3_idx]

def calculate_treatment_cost(disease_info, acres):
    """Calculate treatment cost for given acreage"""
    costs = disease_info.get('cost_per_acre', {'materials': 100, 'labor': 80, 'equipment': 40})
    materials = costs['materials'] * acres
    labor = costs['labor'] * acres
    equipment = costs['equipment'] * acres
    total = materials + labor + equipment
    return {
        'materials': materials,
        'labor': labor,
        'equipment': equipment,
        'total': total
    }

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main application function"""
    load_css()
    
    lang = st.session_state.language
    t = TRANSLATIONS[lang]
    
    # Header with language selector
    col1, col2 = st.columns([9, 1])
    
    with col2:
        with st.container():
            lang_options = ['english', 'hindi', 'bengali']
            lang_display = {'english': 'EN', 'hindi': 'हिं', 'bengali': 'বাং'}
            
            selected_lang = st.selectbox(
                "Language",
                options=lang_options,
                format_func=lambda x: lang_display[x],
                key='lang_select',
                label_visibility='collapsed'
            )
            
            if selected_lang != st.session_state.language:
                st.session_state.language = selected_lang
                st.rerun()
    
    # Hero Section
    st.markdown(f"""
    <div class="hero-section fade-in">
        <div class="hero-title">{t['title']}</div>
        <div class="hero-subtitle">{t['subtitle']}</div>
        <div class="hero-tagline">{t['tagline']}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Load resources
    model = load_model()
    class_names = load_class_names()
    disease_db = load_disease_db()
    crop_calendar = load_crop_calendar()
    
    if not model or not class_names:
        st.error("**System Error**: Unable to load AI model or class definitions. Please ensure all required files are present.")
        st.info("**Required files**: `leaf_guard_best.h5`, `class_names.txt`")
        return
    
    # Stats Banner
    st.markdown('<div class="stats-banner">', unsafe_allow_html=True)
    cols = st.columns(3)
    
    with cols[0]:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">94.37%</div>
            <div class="stat-label">{t['model_accuracy']}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with cols[1]:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{len(class_names)}</div>
            <div class="stat-label">{t['detectable']}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with cols[2]:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{len(st.session_state.scan_history)}</div>
            <div class="stat-label">{t['total_scans']}</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Main Navigation Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        t['scan'],
        t['detectable_diseases'],
        t['cost_calc'],
        t['crop_calendar'],
        t['history']
    ])
    
    # ========================================
    # TAB 1: SCAN & DETECT
    # ========================================
    with tab1:
        st.markdown(f"### {t['scan']}")
        
        col1, col2 = st.columns([1, 1], gap="large")
        
        with col1:
            st.markdown(f"#### {t['upload_image']}")
            uploaded_file = st.file_uploader(
                t['drag_drop'],
                type=['jpg', 'jpeg', 'png'],
                help=t['upload_help']
            )
            
            if uploaded_file:
                image = Image.open(uploaded_file)
                st.image(image, use_container_width=True, caption=t['uploaded_image'])
        
        with col2:
            if uploaded_file:
                st.markdown(f"#### {t['ai_analysis']}")
                
                if st.button(t['analyze_disease'], use_container_width=True, type="primary"):
                    with st.spinner(t['analyzing']):
                        results = predict_disease(model, image, class_names)
                        
                        disease = results[0]['disease']
                        confidence = results[0]['confidence']
                        
                        # Main Result Card
                        st.markdown(f"""
                        <div class="result-card fade-in">
                            <div style="text-align: center;">
                                <div class="disease-badge">{disease.replace('___', ' ').replace('_', ' ').title()}</div>
                            </div>
                            <div class="confidence-meter">
                                <div class="confidence-value">{confidence:.2f}%</div>
                                <div style="color: rgba(255, 235, 205, 0.7); font-size: 1rem; margin-top: 0.5rem; text-transform: uppercase; letter-spacing: 0.08em;">{t['confidence_level']}</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Top 3 Predictions
                        st.markdown(f"### {t['detection_probabilities']}")
                        for i, result in enumerate(results, 1):
                            col_a, col_b = st.columns([3, 1])
                            with col_a:
                                st.write(f"**{i}. {result['disease'].replace('___', ' ').replace('_', ' ').title()}**")
                                st.progress(result['confidence'] / 100)
                            with col_b:
                                st.metric("", f"{result['confidence']:.1f}%")
                        
                        # Get disease information
                        info = get_disease_info(disease, disease_db)
                        
                        # Symptoms
                        st.markdown(f"""
                        <div class="info-section">
                            <div class="section-title">{t['disease_symptoms']}</div>
                            <div class="section-content">{info['symptoms']}</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Treatment Options
                        col_t1, col_t2 = st.columns(2)
                        
                        with col_t1:
                            st.markdown(f"""
                            <div class="info-section">
                                <div class="section-title">{t['organic_treatment']}</div>
                                <div class="section-content">{info.get('organic_treatment', 'Consult agricultural expert')}</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col_t2:
                            st.markdown(f"""
                            <div class="info-section">
                                <div class="section-title">{t['chemical_treatment']}</div>
                                <div class="section-content">{info.get('chemical_treatment', 'Consult agricultural expert')}</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Prevention
                        st.markdown(f"""
                        <div class="info-section">
                            <div class="section-title">{t['prevention_strategies']}</div>
                            <div class="section-content">{info['prevention']}</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Important Facts
                        st.markdown(f"### {t['key_facts']}")
                        facts_cols = st.columns(2)
                        for idx, fact in enumerate(info['facts']):
                            with facts_cols[idx % 2]:
                                st.info(fact)
                        
                        # Save to history
                        buffered = BytesIO()
                        image.save(buffered, format="PNG")
                        img_str = base64.b64encode(buffered.getvalue()).decode()
                        
                        st.session_state.scan_history.insert(0, {
                            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            'image': img_str,
                            'disease': disease,
                            'confidence': confidence
                        })
                        
                        if len(st.session_state.scan_history) > 100:
                            st.session_state.scan_history = st.session_state.scan_history[:100]
                        
                        st.success(t['analysis_complete'])
            else:
                st.info(t['upload_to_begin'])
                
                st.markdown(f"### {t['upload_guidelines']}")
                guidelines = [
                    t['guideline_1'],
                    t['guideline_2'],
                    t['guideline_3'],
                    t['guideline_4'],
                    t['guideline_5']
                ]
                
                for guideline in guidelines:
                    st.success(guideline)
    
    # ========================================
    # TAB 2: DETECTABLE DISEASES
    # ========================================
    with tab2:
        st.markdown(f"### {t['disease_database']}")
        
        st.markdown(f"""
        <div class="info-section">
            <div class="section-content">{t['database_desc']}</div>
        </div>
        """, unsafe_allow_html=True)
        
        for crop, diseases in DETECTABLE_DISEASES.items():
            st.markdown(f"### {t[crop.lower()]}")
            
            cols = st.columns(2)
            for idx, disease in enumerate(diseases):
                with cols[idx % 2]:
                    severity_class = f"severity-{disease['severity'].lower()}"
                    
                    st.markdown(f"""
                    <div class="disease-list">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <strong style="font-size: 1.05rem;">{disease['name']}</strong>
                            <span class="severity-badge {severity_class}">{t[f'severity_{disease["severity"].lower()}']}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown(f"### {t['best_practices']}")
        
        best_practices = [
            (t['lighting'], t['lighting_desc']),
            (t['focus'], t['focus_desc']),
            (t['distance'], t['distance_desc']),
            (t['angle'], t['angle_desc']),
            (t['background'], t['background_desc'])
        ]
        
        cols = st.columns(2)
        for idx, (title, description) in enumerate(best_practices):
            with cols[idx % 2]:
                st.markdown(f"""
                <div class="info-section">
                    <div class="section-title">{title}</div>
                    <div class="section-content">{description}</div>
                </div>
                """, unsafe_allow_html=True)
    
    # ========================================
    # TAB 3: COST CALCULATOR
    # ========================================
    with tab3:
        st.markdown(f"### {t['cost_calculator']}")
        
        col1, col2 = st.columns([2, 1], gap="large")
        
        with col1:
            st.markdown(f"#### {t['input_parameters']}")
            
            disease_options = [k for k in disease_db.keys() if k != 'default']
            selected_disease = st.selectbox(
                t['select_disease'],
                disease_options,
                format_func=lambda x: x.replace('_', ' ').title(),
                key='cost_disease_select'
            )
            
            acres = st.number_input(
                t['affected_area'],
                min_value=0.1,
                max_value=1000.0,
                value=1.0,
                step=0.5,
                help=t['area_help'],
                key='cost_acres'
            )
            
            treatment_type = st.radio(
                t['treatment_preference'],
                [t['organic'], t['chemical'], t['integrated']],
                horizontal=True,
                help=t['treatment_help'],
                key='cost_treatment_type'
            )
            
            if st.button(t['calculate_cost'], use_container_width=True, type="primary", key='calc_cost_btn'):
                info = disease_db[selected_disease]
                costs = calculate_treatment_cost(info, acres)
                
                # Adjust costs based on treatment type
                if treatment_type == t['organic']:
                    costs['materials'] *= 0.85
                elif treatment_type == t['chemical']:
                    costs['materials'] *= 1.15
                
                costs['total'] = costs['materials'] + costs['labor'] + costs['equipment']
                
                # Display Results
                st.markdown(f"""
                <div class="result-card fade-in">
                    <h3 style="text-align: center; margin-bottom: 1.25rem;">{t['cost_analysis']} {acres} {t['acres']}</h3>
                    <div class="confidence-meter">
                        <div class="confidence-value">₹{costs['total']:,.0f}</div>
                        <div style="color: rgba(255, 235, 205, 0.7); font-size: 1rem; margin-top: 0.5rem;">
                            {t['total_cost']}
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Cost Breakdown
                st.markdown(f"### {t['cost_breakdown']}")
                cols_cost = st.columns(3)
                
                with cols_cost[0]:
                    st.markdown(f"""
                    <div class="stat-card">
                        <div class="stat-value">₹{costs['materials']:,.0f}</div>
                        <div class="stat-label">{t['materials']}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with cols_cost[1]:
                    st.markdown(f"""
                    <div class="stat-card">
                        <div class="stat-value">₹{costs['labor']:,.0f}</div>
                        <div class="stat-label">{t['labor']}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with cols_cost[2]:
                    st.markdown(f"""
                    <div class="stat-card">
                        <div class="stat-value">₹{costs['equipment']:,.0f}</div>
                        <div class="stat-label">{t['equipment']}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Treatment Details
                st.markdown(f"### {t['treatment_protocol']}")
                st.markdown(f"""
                <div class="info-section">
                    <div class="section-title">{t['selected_treatment']} {treatment_type}</div>
                    <div class="section-content">
                        <strong>{t['recommended_products']}</strong><br>
                        {info.get(f'{selected_disease.lower()}_treatment', info.get('organic_treatment', 'Consult agricultural expert'))}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                st.info(t['cost_note'])
        
        with col2:
            st.markdown(f"### {t['cost_components']}")
            
            components = [
                (t['materials'], [t['materials_list'], t['copper_compounds'], t['organic_treatments'], t['protective_equipment']]),
                (t['labor'], [t['application_time'], t['plant_removal'], t['field_monitoring'], t['post_treatment']]),
                (t['equipment'], [t['sprayers'], t['hand_tools'], t['safety_gear'], t['storage']])
            ]
            
            for title, items in components:
                st.markdown(f"""
                <div class="info-section">
                    <div class="section-title">{title}</div>
                    <div class="section-content">
                        {"<br>".join([f"• {item}" for item in items])}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown(f"### {t['savings_tips']}")
            st.success(t['tip_1'])
            st.success(t['tip_2'])
            st.success(t['tip_3'])
            st.success(t['tip_4'])
    
    # ========================================
    # TAB 4: CROP CALENDAR
    # ========================================
    with tab4:
        st.markdown(f"### {t['crop_planning']}")
        
        crops = list(crop_calendar.keys())
        
        selected_crop = st.selectbox(
            t['select_crop'],
            crops,
            format_func=lambda x: t[x],
            key='calendar_crop_select'
        )
        
        crop_info = crop_calendar[selected_crop]
        
        # Crop Overview Header
        st.markdown(f"""
        <div class="feature-card">
            <h2 style="color: #ffb74d; margin-bottom: 0.5rem;">{t[selected_crop]}</h2>
            <div style="color: rgba(255, 235, 205, 0.7); font-size: 1rem; margin-bottom: 1.5rem;">{t['growing_guide']}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Crop Information Grid
        st.markdown('<div class="crop-info-grid">', unsafe_allow_html=True)
        
        info_items = [
            (t['planting_season'], crop_info['planting']),
            (t['harvest_season'], crop_info['harvest']),
            (t['optimal_temp'], crop_info['temp']),
            (t['water_requirements'], crop_info['water']),
            (t['plant_spacing'], crop_info['spacing']),
            (t['soil_ph'], crop_info['soil_ph']),
            (t['days_to_harvest'], crop_info['days_to_harvest']),
            (t['best_season'], crop_info['season'])
        ]
        
        for label, value in info_items:
            st.markdown(f"""
            <div class="crop-info-item">
                <div class="crop-info-label">{label}</div>
                <div class="crop-info-value">{value}</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Current Month Recommendations
        current_month = datetime.now().strftime("%B")
        st.markdown(f"""
        <div class="info-section" style="margin-top: 1.5rem;">
            <div class="section-title">{current_month} - {t['monthly_action']}</div>
            <div class="section-content">
                <strong>{t['planting']}</strong> {t['planting_desc']}<br>
                <strong>{t['maintenance']}</strong> {t['maintenance_desc']}<br>
                <strong>{t['soil_prep']}</strong> {t['soil_prep_desc']}<br>
                <strong>{t['fertilization']}</strong> {t['fertilization_desc']}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # ========================================
    # TAB 5: HISTORY & ANALYTICS
    # ========================================
    with tab5:
        st.markdown(f"### {t['scan_history']}")
        
        if len(st.session_state.scan_history) == 0:
            st.markdown(f"""
            <div class="info-section" style="text-align: center; padding: 2.5rem;">
                <div class="section-title">{t['no_history']}</div>
                <div class="section-content">{t['no_history_desc']}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Summary Statistics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(t['total_scans'], len(st.session_state.scan_history))
            
            with col2:
                avg_confidence = sum([s['confidence'] for s in st.session_state.scan_history]) / len(st.session_state.scan_history)
                st.metric(t['avg_confidence'], f"{avg_confidence:.1f}%")
            
            with col3:
                if st.button(t['clear_history'], use_container_width=True, key='clear_history_btn'):
                    st.session_state.scan_history = []
                    st.success(t['history_cleared'])
                    st.rerun()
            
            # Detailed Scan History
            st.markdown(f"### {t['scan_records']}")
            
            for idx, scan in enumerate(st.session_state.scan_history[:20]):
                col_img, col_info = st.columns([1, 2])
                
                with col_img:
                    try:
                        img_data = base64.b64decode(scan['image'])
                        img = Image.open(BytesIO(img_data))
                        st.image(img, use_container_width=True)
                    except:
                        st.info("Image preview unavailable")
                
                with col_info:
                    st.markdown(f"""
                    <div class="feature-card">
                        <div style="font-size: 1.3rem; font-weight: 700; color: #ffb74d; margin-bottom: 0.65rem;">
                            {scan['disease'].replace('___', ' ').replace('_', ' ').title()}
                        </div>
                        <div style="display: flex; gap: 1rem; margin: 0.85rem 0; flex-wrap: wrap;">
                            <div style="background: rgba(255, 183, 77, 0.2); padding: 0.45rem 0.95rem; border-radius: 8px; border: 1px solid rgba(255, 183, 77, 0.4);">
                                <strong>{t['confidence']}</strong> {scan['confidence']:.2f}%
                            </div>
                        </div>
                        <div style="color: #ffb74d; font-weight: 600; margin-top: 0.85rem; font-size: 0.95rem;">
                            {scan['timestamp']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                if idx < len(st.session_state.scan_history) - 1:
                    st.markdown("---")
            
            if len(st.session_state.scan_history) > 20:
                st.info(f"{t['showing_latest']} {len(st.session_state.scan_history)} {t['total_scans_text']}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 1.75rem; color: rgba(255, 235, 205, 0.6);">
        <div style="font-size: 1rem; font-weight: 600; margin-bottom: 0.4rem;">Leaf Guard AI</div>
        <div style="font-size: 0.85rem;">Empowering Farmers with AI-Driven Precision Agriculture</div>
        <div style="margin-top: 0.85rem; font-size: 0.8rem;">
            Model Accuracy: 94.37% | Powered by TensorFlow & MobileNetV2
        </div>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()