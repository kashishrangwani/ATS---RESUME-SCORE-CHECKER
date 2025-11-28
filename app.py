import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fpdf import FPDF
import matplotlib.pyplot as plt
from PyPDF2 import PdfReader
import joblib

# ---------- LOAD TRAINED ML ARTIFACTS ----------
# Ensure these files exist: resume_model.pkl and tfidf_vectorizer.pkl
model = joblib.load("resume_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# ---------------------------
# Default Skills
# ---------------------------
DEFAULT_SKILLS = [
    "python","machine learning","data analysis","pandas","numpy",
    "scikit-learn","sql","git","streamlit","flask","django",
    "javascript","html","css","tensorflow","keras","nlp"
]

# ---------------------------
# Helper Functions
# ---------------------------
def clean_text(text):
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_skills(text, skills_list):
    matched = [skill for skill in skills_list if skill in text]
    missing = [skill for skill in skills_list if skill not in text]
    return matched, missing

def compute_skill_score(resume_text, jd_text, skills_list):
    matched, missing = extract_skills(resume_text, skills_list)
    score = len(matched)/max(len(skills_list),1)
    return score, matched, missing

def compute_keyword_similarity(resume_text, jd_text):
    vectorizer_local = TfidfVectorizer()
    vectors = vectorizer_local.fit_transform([resume_text, jd_text])
    sim = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
    return sim

def compute_education_score(resume_text, jd_text):
    return 1.0, "Bachelor", "Bachelor"

def compute_resume_quality(resume_text):
    return 1.0, {'contact_present': True, 'bullets': resume_text.count('-')}

# ---------- ML helper functions ----------
def ml_predict_label(text):
    """Return predicted class label for given text."""
    if not text:
        return "Unknown"
    X = vectorizer.transform([text])
    pred = model.predict(X)[0]
    return pred

def ml_predict_proba(text):
    """Return probability of predicted class (0..1). If model doesn't support proba, return None."""
    if not text:
        return None
    X = vectorizer.transform([text])
    try:
        probs = model.predict_proba(X)[0]
        # get probability for predicted class
        classes = model.classes_
        pred = model.predict(X)[0]
        # find index
        idx = list(classes).index(pred)
        return float(probs[idx])
    except Exception:
        return None

# PDF Report Class
class PDFReport(FPDF):
    def __init__(self, title='Report'):
        super().__init__()
        self.title = title
    def header(self):
        self.set_font('Arial','B',16)
        self.set_text_color(11,61,145)
        self.cell(0,10,self.title,ln=True,align='C')
        self.ln(10)
    def build(self,data):
        self.add_page()
        self.set_font('Arial','',12)
        self.multi_cell(0,6,data['summary'])
        self.ln(5)
        self.multi_cell(0,6,'Scores:')
        for k,v in data['scores'].items():
            self.cell(0,6,f'{k}: {v:.2f}',ln=True)
        self.ln(5)
        self.multi_cell(0,6,'Matched Skills: ' + ', '.join(data['matched_skills']))
        self.multi_cell(0,6,'Missing Skills: ' + ', '.join(data['missing_skills']))
        if data['suggestions']:
            self.ln(5)
            self.multi_cell(0,6,'Suggestions:')
            for s in data['suggestions']:
                self.multi_cell(0,6,f'- {s}')

# ---------------------------
# Page Functions
# ---------------------------
def run_single_resume_page():
    st.title('📄 Single Resume Analysis')

    use_custom_skills = st.sidebar.checkbox('Use custom skills list', value=False)
    if use_custom_skills:
        skills_input = st.sidebar.text_area('Enter comma-separated skills', height=200)
        skills_list = [s.strip().lower() for s in skills_input.split(',') if s.strip()]
    else:
        skills_list = DEFAULT_SKILLS

    col1, col2 = st.columns([1,2])
    with col1:
        uploaded_file = st.file_uploader('Upload Resume (PDF)', type=['pdf'])
        resume_text_manual = st.text_area('Or paste resume text', height=200)
        analyze_button = st.button('Analyze')
    with col2:
        jd_text = st.text_area('Paste Job Description here', height=450)

    if analyze_button:
        resume_text = ''
        if uploaded_file:
            try:
                reader = PdfReader(uploaded_file)
                resume_text = ' '.join([page.extract_text() for page in reader.pages if page.extract_text()])
            except Exception:
                st.error('Error reading PDF')
        if not resume_text and resume_text_manual:
            resume_text = resume_text_manual

        resume_text = clean_text(resume_text)
        jd_text = clean_text(jd_text)

        if not resume_text or not jd_text:
            st.error('Please provide both Resume and Job Description')
        else:
            # compute existing scores
            skill_score_val, matched_skills, missing_skills = compute_skill_score(resume_text,jd_text,skills_list)
            keyword_sim = compute_keyword_similarity(resume_text,jd_text)
            education_score_val, _, _ = compute_education_score(resume_text,jd_text)
            resume_quality_val, quality_checks = compute_resume_quality(resume_text)
            final_score = (0.4*skill_score_val + 0.3*keyword_sim + 0.1*education_score_val + 0.2*resume_quality_val)*100

            # ML prediction
            ml_label = ml_predict_label(resume_text)
            ml_prob = ml_predict_proba(resume_text)

            # Dashboard layout
            left_col, right_col = st.columns([1,1.2])
            labels = np.array(['Skills','Keywords','Education','Resume Quality'])
            stats = np.array([skill_score_val*100, keyword_sim*100, education_score_val*100, resume_quality_val*100])
            angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False)
            stats_loop = np.concatenate((stats,[stats[0]]))
            angles_loop = np.concatenate((angles,[angles[0]]))

            with left_col:
                st.subheader('Score Breakdown')
                fig = plt.figure(figsize=(3.5,3.5))
                ax = fig.add_subplot(111, polar=True)
                ax.plot(angles_loop, stats_loop, 'o-', linewidth=2, color='#1f77b4')
                ax.fill(angles_loop, stats_loop, alpha=0.25, color='#1f77b4')
                ax.set_thetagrids(angles*180/np.pi, labels)
                ax.set_ylim(0,100)
                ax.tick_params(labelsize=10)
                ax.grid(color='gray', linestyle='--', linewidth=0.5)
                plt.tight_layout()
                st.pyplot(fig)

            with right_col:
                st.subheader('✅ Matched Skills')
                st.write(', '.join(matched_skills) if matched_skills else 'No skills matched.')

                st.subheader('❌ Missing Skills')
                st.write(', '.join(missing_skills) if missing_skills else 'All required skills are present!')

                st.subheader('💡 Suggestions')
                suggestions=[]
                if missing_skills:
                    suggestions.append('Consider adding: '+', '.join(missing_skills[:15]))
                if not quality_checks['contact_present']:
                    suggestions.append('Add contact info clearly at the top.')
                if quality_checks['bullets']<3:
                    suggestions.append('Add bullet points for better readability.')
                if suggestions:
                    for s in suggestions:
                        st.info(s)
                else:
                    st.success('Your resume is strong!')

            st.markdown("<h3 style='color:#0b3d91;'>Overall Score</h3>", unsafe_allow_html=True)
            st.metric('Final ATS Score', f'{round(final_score,2)} / 100')

            # ML display
            st.subheader("🤖 ML Model Prediction")
            if ml_prob is not None:
                st.write(f"**Predicted category:** {ml_label} (confidence: {ml_prob*100:.1f}%)")
            else:
                st.write(f"**Predicted category:** {ml_label}")

            # PDF download
            pdf = PDFReport(title='ATS Resume Analysis')
            data = {
                'summary': f'ATS Score: {round(final_score,2)} / 100\\nML Prediction: {ml_label}',
                'scores': {'Skills': skill_score_val*100, 'Keywords': keyword_sim*100,
                           'Education': education_score_val*100, 'Resume Quality': resume_quality_val*100},
                'matched_skills': matched_skills,
                'missing_skills': missing_skills,
                'suggestions': suggestions
            }
            pdf.build(data)
            pdf_bytes = pdf.output(dest='S').encode('latin-1')
            st.download_button('Download PDF', data=pdf_bytes, file_name='ATS_Report.pdf')

def run_bulk_analysis_page():
    st.title('📂 Bulk Resume Analysis')

    uploaded_files = st.file_uploader("Upload multiple resumes", type=['pdf'], accept_multiple_files=True)
    jd_text = st.text_area("Paste Job Description here", height=200)
    analyze_button = st.button("Analyze All")

    if analyze_button and uploaded_files and jd_text:
        results = []
        jd_text_clean = clean_text(jd_text)

        for file in uploaded_files:
            try:
                reader = PdfReader(file)
                resume_text = ' '.join([page.extract_text() for page in reader.pages if page.extract_text()])
            except Exception:
                resume_text = ""
            resume_text_clean = clean_text(resume_text)

            skill_score, matched, missing = compute_skill_score(resume_text_clean, jd_text_clean, DEFAULT_SKILLS)
            keyword_sim = compute_keyword_similarity(resume_text_clean, jd_text_clean)
            education_score, _, _ = compute_education_score(resume_text_clean, jd_text_clean)
            resume_quality, _ = compute_resume_quality(resume_text_clean)
            final_score = (0.4*skill_score + 0.3*keyword_sim + 0.1*education_score + 0.2*resume_quality)*100

            # ML prediction for each resume
            ml_label = ml_predict_label(resume_text_clean)
            ml_prob = ml_predict_proba(resume_text_clean)
            ml_prob_pct = round(ml_prob*100,1) if ml_prob is not None else None

            results.append({
                "Resume": file.name,
                "Final Score": round(final_score,2),
                "Skills Matched": len(matched),
                "Keywords Similarity": round(keyword_sim*100,2),
                "ML Label": ml_label,
                "ML Prob (%)": ml_prob_pct
            })

        df_results = pd.DataFrame(results)
        st.subheader("Comparative Analysis")
        st.dataframe(df_results)
        st.bar_chart(df_results.set_index('Resume')['Final Score'])

        # Download Excel report (includes ML columns)
        df_results.to_excel('comparative_analysis.xlsx', index=False)
        st.download_button('Download Comparative Report', data=open('comparative_analysis.xlsx','rb'), file_name='Comparative_Report.xlsx')

# ---------------------------
# Main Navigation
# ---------------------------
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choose Page", ["Single Resume Analysis", "Bulk Resume Analysis"])

if page == "Single Resume Analysis":
    run_single_resume_page()
else:
    run_bulk_analysis_page()
