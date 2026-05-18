"""
cases.py — Clinical test cases for Single-Agent vs Multi-Agent evaluation.

Design principles:
  - Every case contains at least one red herring or misleading finding
  - Every case has at least two patient-specific safety considerations
  - Every case requires ruling out at least one life-threatening alternative
  - Cases span multiple specialties to prevent architecture over-fitting

Cases 1-3: Original benchmark (cardiology / elderly multimorbidity / obstetrics)
Cases 4-8: New cases (paediatrics / neurology / GYN emergency / metabolic / rheumatology)
"""

CLINICAL_CASES = [
    # ── ORIGINAL CASES ────────────────────────────────────────────────────────
    {
        "id": 1,
        "title": "Atypical Chest Pain with Confounding History",
        "specialty": "Cardiology / Emergency",
        "case": """
Patient: 52-year-old female
Chief Complaint: Chest discomfort and fatigue for 6 hours. Describes it as "indigestion-like pressure."
Vitals: BP 138/88 (R arm) / 112/76 (L arm), HR 96, RR 20, O2 Sat 96%, Temp 37.2°C
Symptoms: Mild chest pressure radiating to jaw and back, nausea, diaphoresis, no classic crushing pain.
          She initially attributed it to spicy food she ate. Mild shortness of breath on exertion.
History: Hypertension, GERD (on omeprazole), Type 2 Diabetes, BMI 34. No prior cardiac history.
         Family history: Father died of "stomach problems" at 56 (actual cause unknown).
ECG: Non-specific ST changes in V4-V6. No clear STEMI pattern. Possible new T-wave inversions.
Labs: Troponin I = 0.07 ng/mL (borderline, normal <0.04). BNP 420 pg/mL (elevated).
      Lipase slightly elevated at 180 U/L. D-dimer 1.1 µg/mL (mildly elevated).
Imaging: CXR shows mild cardiomegaly. No obvious pneumothorax.
        """,
        "expected_keywords": [
            "NSTEMI", "ACS", "serial troponin", "heparin", "aspirin", "cardiology",
            "aortic dissection", "pulmonary embolism", "repeat ECG", "differential"
        ],
    },
    {
        "id": 2,
        "title": "Altered Mental Status in Elderly — Multiple Competing Diagnoses",
        "specialty": "Internal Medicine / Emergency",
        "case": """
Patient: 78-year-old male, brought from nursing home
Chief Complaint: Acute confusion and agitation since this morning. Nursing staff say "he's not himself."
Vitals: BP 96/58, HR 112 (irregular), RR 22, Temp 38.6°C, O2 Sat 91% on room air, GCS 12/15
Symptoms: Confusion, pulling at IV lines, intermittent agitation. Not responding to name.
          Noted to have urinary incontinence (new). Mild abdominal distension.
History: Atrial fibrillation (on warfarin, INR today = 3.8 — supratherapeutic),
         Hypertension, CKD Stage 3, Benign Prostatic Hyperplasia, recent 7-day course of
         ciprofloxacin for UTI completed 3 days ago. Lives in nursing home, baseline
         cognitively intact. Family reports he was fine yesterday.
Labs: WBC 18,500 (elevated, with 88% neutrophils), CRP 210 mg/L, Creatinine 2.8 (baseline 1.6),
      Urine: cloudy, nitrite +, WBC >50/hpf. Blood cultures x2 pending.
      Na 131, K 5.2, Glucose 188. INR 3.8. Lactate 3.1 mmol/L.
ECG: Atrial fibrillation with rapid ventricular rate at 118 bpm. No acute ischemic changes.
        """,
        "expected_keywords": [
            "sepsis", "urosepsis", "delirium", "warfarin", "INR reversal",
            "atrial fibrillation", "rate control", "blood cultures", "IV antibiotics",
            "ICU", "fluid resuscitation", "AKI", "lactate"
        ],
    },
    {
        "id": 3,
        "title": "Post-Partum Breathlessness — Diagnostic Trap",
        "specialty": "Obstetrics / Cardiology",
        "case": """
Patient: 29-year-old female, 8 days post-partum (vaginal delivery)
Chief Complaint: Progressive shortness of breath over 3 days. Initially dismissed as "new mother exhaustion."
Vitals: BP 148/96, HR 118, RR 26, O2 Sat 93% on room air, Temp 37.3°C
Symptoms: Worsening dyspnea even at rest, orthopnea (needs 3 pillows to sleep), bilateral leg swelling,
          palpitations. Mild chest tightness. Denies cough or fever.
          Feels extremely fatigued — says she "can't carry the baby anymore."
History: Uncomplicated pregnancy. No prior cardiac history.
         BMI 28. Gestational hypertension during third trimester — resolved at delivery per records.
         No DVT prophylaxis given post-partum (not prescribed). No family history of cardiac disease.
Labs: BNP 1840 pg/mL (severely elevated). Troponin 0.18 ng/mL (elevated).
      D-dimer 4.2 µg/mL (significantly elevated). CBC: Hb 9.8 g/dL, WBC normal, platelets 142.
      TSH 5.8 mIU/L (mildly elevated). CRP 38 mg/L.
ECG: Sinus tachycardia. Non-specific T-wave changes. No STEMI pattern.
CXR: Bilateral pulmonary infiltrates, cardiomegaly, blunting of costophrenic angles.
Echo (preliminary report): Severely reduced EF ~25-30%, global hypokinesia, dilated ventricles.
        """,
        "expected_keywords": [
            "peripartum cardiomyopathy", "heart failure", "pulmonary embolism",
            "bromocriptine", "diuretics", "echocardiogram", "anticoagulation",
            "ICU", "cardiology", "EF", "preeclampsia", "differential"
        ],
    },

    # ── NEW CASES ─────────────────────────────────────────────────────────────
    {
        "id": 4,
        "title": "Paediatric Fever with Petechial Rash — Diagnostic Emergency",
        "specialty": "Paediatrics / Emergency",
        "case": """
Patient: 8-year-old male, brought by parents
Chief Complaint: High fever and spreading rash since last night. Parents initially thought it was "an allergic reaction."
Vitals: BP 88/54, HR 148, RR 32, Temp 39.8°C, O2 Sat 95% on room air, GCS 13/15 (confused, slow to respond)
Symptoms: Non-blanching petechial rash beginning on lower limbs, now spreading to trunk.
          Neck stiffness on examination. Photophobia. Headache reported before confusion onset.
          Vomited twice. No recent travel. Two classmates reportedly had a "flu-like illness" last week.
History: Up to date on vaccinations including MenB and MenACWY. No known allergies.
         No current medications. No immunodeficiency history.
Labs: WBC 22,000 (88% neutrophils), CRP 310 mg/L, Procalcitonin 18 ng/mL.
      Platelets 68,000 (low). Fibrinogen 1.1 g/L (low). PT prolonged.
      Blood glucose 3.1 mmol/L. Na 129 (low). Lactate 4.2 mmol/L.
      Blood cultures x2 drawn but pending.
Imaging: CXR clear. CT head not yet performed.
Notes: Rash appeared ~8 hours ago and has visibly progressed over the past 2 hours.
       Lumbar puncture not yet performed (child increasingly unwell).
        """,
        "expected_keywords": [
            "meningococcaemia", "meningococcal septicaemia", "bacterial meningitis",
            "IV ceftriaxone", "benzylpenicillin", "DIC", "septic shock",
            "ICU", "fluid resuscitation", "lumbar puncture", "contraindication",
            "CT head", "dexamethasone", "differential", "ITP"
        ],
    },
    {
        "id": 5,
        "title": "Transient Focal Neurology in an Anticoagulated Patient",
        "specialty": "Neurology / Emergency",
        "case": """
Patient: 67-year-old male
Chief Complaint: Right-sided arm weakness and slurred speech lasting ~90 minutes, now largely resolved.
Vitals: BP 174/98, HR 84 (regular), RR 16, O2 Sat 98%, Temp 36.9°C
Symptoms: Right arm weakness (resolved ~80% at presentation), mild facial droop (largely resolved),
          mild expressive dysphasia still present. Patient also reports a generalised headache during the episode,
          which he describes as "the worst headache of his life." Headache now 4/10.
History: Paroxysmal atrial fibrillation (on apixaban 5mg BD — last dose taken this morning).
         Hypertension (on amlodipine), Type 2 Diabetes (on metformin). Ex-smoker.
         CHADS2-VASc score 5. No prior stroke or TIA. No recent falls or head trauma.
Labs: INR 1.1 (on DOAC — not interpretable). Glucose 9.2. Creatinine 98 µmol/L.
      CBC normal. Coagulation screen: PT 14.2s, APTT 42s (mildly elevated — DOAC effect).
      Troponin I 0.03 ng/mL (borderline).
Imaging: CT head (non-contrast): No acute haemorrhage. Possible hyperdense MCA sign on left — radiologist to confirm.
         MRI brain not yet performed.
ECG: Sinus rhythm. No acute ischaemic changes. No P-wave abnormality currently visible.
        """,
        "expected_keywords": [
            "TIA", "ischaemic stroke", "subarachnoid haemorrhage", "MRI brain",
            "MRA", "ABCD2 score", "anticoagulation", "apixaban", "thrombolysis",
            "tPA", "contraindication", "CT angiography", "neurology", "atrial fibrillation",
            "carotid imaging", "dual antiplatelet", "DOAC", "differential"
        ],
    },
    {
        "id": 6,
        "title": "Acute Lower Abdominal Pain in a Young Female — Haemodynamic Instability",
        "specialty": "Gynaecology / Emergency",
        "case": """
Patient: 23-year-old female
Chief Complaint: Sudden-onset severe lower abdominal pain with a syncopal episode 30 minutes ago.
Vitals: BP 84/52 (lying), HR 132, RR 24, Temp 37.1°C, O2 Sat 97%
Symptoms: Sharp, constant lower abdominal pain radiating to right shoulder tip.
          One episode of syncope in the ambulance. Nausea, no vomiting.
          Last menstrual period approximately 6 weeks ago — patient says cycle is "always irregular."
          Mild vaginal spotting noted since yesterday; attributed to "late period."
          Denies recent sexual activity (partner away for 3 weeks), but reports prior unprotected intercourse.
History: No prior pregnancies. No known gynaecological history.
         Uses no regular contraception. No current medications. No known allergies.
         Appendicectomy at age 12. Previous episode of pelvic pain 1 year ago — not investigated.
Labs: Hb 8.2 g/dL (acutely low — baseline unknown). WBC 11,200. CRP 28 mg/L.
      Serum beta-hCG: 4,200 mIU/mL (POSITIVE — patient unaware she was pregnant).
      Progesterone 6 nmol/L (low for gestational age, concerning for non-viable pregnancy).
      Blood type O negative (not yet cross-matched).
Imaging: FAST ultrasound: Free fluid in Morrison's pouch and pelvis. No intrauterine pregnancy visualised.
         Adnexal mass 3.2cm on the right — indeterminate on FAST.
        """,
        "expected_keywords": [
            "ectopic pregnancy", "ruptured ectopic", "haemoperitoneum", "beta-hCG",
            "surgical emergency", "laparoscopy", "salpingectomy",
            "ovarian torsion", "appendicitis", "differential",
            "IV access", "fluid resuscitation", "cross-match", "anti-D immunoglobulin",
            "Rh negative", "gynaecology", "FAST", "progesterone"
        ],
    },
    {
        "id": 7,
        "title": "Confusion and Vomiting in a Known Diabetic — Metabolic Emergency",
        "specialty": "Endocrinology / Emergency",
        "case": """
Patient: 44-year-old male with Type 1 Diabetes, brought in by flatmate
Chief Complaint: Found confused at home with vomiting. Has been "unwell for 2 days."
Vitals: BP 102/64, HR 122, RR 28 (deep, sighing), Temp 37.6°C, O2 Sat 97%, GCS 12/15
Symptoms: Profuse vomiting (×5 today), severe abdominal pain, fruity breath odour.
          Confusion — unable to give coherent history. Flatmate reports he ran out of insulin 3 days ago.
          Also reports patient drinks "heavily" — approximately 10-14 units of alcohol per day.
          Last meal unclear. Reports he has "been cutting down" his insulin to lose weight.
History: Type 1 Diabetes (on basal-bolus insulin — currently without medication).
         Known alcohol use disorder. No other documented comorbidities.
         No current medications apart from insulin (ran out). No known allergies.
Labs: Glucose 38.2 mmol/L. Ketones (blood): 6.8 mmol/L (severe).
      pH 7.08 (severe acidosis). HCO3 8 mmol/L. pCO2 18 mmHg (compensatory).
      Na 128 (corrected Na 136). K 6.1 mmol/L (appears hyperkalaemic but total body K depleted).
      Creatinine 188 µmol/L (likely pre-renal). Lactate 3.8 mmol/L.
      Phosphate 0.6 mmol/L (low). Magnesium 0.6 mmol/L (low).
      Blood alcohol level: 42 mg/dL (mildly elevated). Lipase 210 U/L (mildly elevated).
      Thiamine not yet measured.
ECG: Sinus tachycardia. Peaked T waves (hyperkalaemia). No QTc prolongation currently.
        """,
        "expected_keywords": [
            "DKA", "diabetic ketoacidosis", "insulin infusion", "fluid resuscitation",
            "potassium", "hyperkalaemia", "phosphate", "cerebral oedema",
            "Wernicke encephalopathy", "thiamine", "alcohol", "differential",
            "HHS", "lactic acidosis", "monitoring", "ICU", "insulin omission",
            "diabulimia", "magnesium"
        ],
    },
    {
        "id": 8,
        "title": "Acute Monoarthritis in an Immunocompromised Patient",
        "specialty": "Rheumatology / Infectious Disease",
        "case": """
Patient: 61-year-old female with Rheumatoid Arthritis
Chief Complaint: Acutely swollen, red, painful right knee since yesterday. Unable to weight-bear.
Vitals: BP 128/80, HR 96, RR 18, Temp 38.1°C, O2 Sat 99% on room air
Symptoms: Right knee: hot, erythematous, markedly tender, large effusion — acute onset.
          Reports a single episode of similar knee swelling 2 years ago ("gout attack" per GP).
          Currently on methotrexate 20mg/week and prednisone 10mg/day (for RA).
          No recent trauma. No recent urogenital symptoms. No diarrhoeal illness.
          Recently completed a 5-day course of amoxicillin for a dental procedure 2 weeks ago.
History: Rheumatoid Arthritis (seropositive, RF+, anti-CCP+). Hypertension (on lisinopril).
         CKD Stage 2. Previous "gout" per GP records — never formally confirmed with joint aspiration.
Labs: WBC 14,200 (elevated). CRP 188 mg/L. ESR 96 mm/hr.
      Uric acid: 0.42 mmol/L (mildly elevated, normal <0.36 in females).
      Creatinine 132 µmol/L (slightly above baseline of 112).
      Synovial fluid (aspirated in ED): WBC 92,000 cells/µL (88% neutrophils) — highly inflammatory.
      Synovial fluid Gram stain: Gram-positive cocci in clusters (preliminary — culture pending).
      Crystals: No crystals identified on polarised microscopy.
Blood cultures: x2 drawn, pending.
Imaging: Knee X-ray: Moderate joint space narrowing (RA changes). No fracture. Soft tissue swelling.
        """,
        "expected_keywords": [
            "septic arthritis", "staphylococcus", "IV antibiotics", "flucloxacillin",
            "vancomycin", "joint washout", "surgical drainage", "orthopaedics",
            "gout", "pseudogout", "methotrexate", "immunosuppression",
            "hold immunosuppressants", "contraindication", "crystal arthropathy",
            "differential", "blood cultures", "synovial fluid", "reactive arthritis"
        ],
    },
]
