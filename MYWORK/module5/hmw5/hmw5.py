import pickle


with open('/workspaces/ML-zoomcamp/MYWORK/module5/hmw5/pipeline.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)


X = dv.transform(
    {
  "lead_source": "paid_ads",
  "industry": "technology",
  "employment_status": "employed",
  "location": "north_america",
  "number_of_courses_viewed": 2,
  "annual_income": 79276.0,
  "interaction_count": 4,
  "lead_score": 0.41
}
)


p = model.predict_proba(X)[:, 1]



print(round(0.769799, 3))