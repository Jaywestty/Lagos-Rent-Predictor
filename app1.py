import streamlit as st
import joblib
import pandas as pd

# Load model
with open('Regressor.pkl', 'rb') as f:
    model = joblib.load(f)

# Score map from original logic
score_map = {
    5: ['banana island', 'bourdillon', 'victoria island extension'],
    4: [
        'lekki phase 1', 'old ikoyi', 'osborne foreshore estate', 'parkview estate',
        'dolphin estate', 'victoria island', '1004', 'ikoyi', 'oniru',
        'adeola odeku', 'ligali ayorinde', 'awolowo road'
    ],
    3: [
        'chevron', 'lekki scheme', 'other lekki', 'agungi', 'osapa london', 'ologolo',
        'ajah', 'ikate', 'ikota', 'ado', 'sangotedo', 'badore',
        'allen avenue', 'opebi', 'oregun'
    ],
    2: [
        'yaba', 'other yaba', 'gbagada', 'other gbagada', 'ikeja', 'other ikeja',
        'surulere', 'other surulere', 'toyin street', 'adeniyi jones',
        'alausa', 'awolowo way', 'omole phase 1', 'omole phase 2',
        'magodo gra phase 1', 'magodo', 'ojodu berger', 'other ojodu',
        'isheri north', 'bode thomas', 'randle avenue', 'onike'
    ],
    1: [
        'oworonshoki', 'lawanson', 'ijesha', 'adelabu', 'aguda', 'ogunlana',
        'marsha', 'ojuelegba', 'ebute-metta', 'sabo', 'alagomeji', 'jibowu',
        'akoka', 'ifako', 'soluyi', 'medina', 'millennium ups'
    ]
}

# Build lookup and list of neighborhoods
neighborhood_list = sorted([n.title() for group in score_map.values() for n in group])
neighborhood_lookup = {n.title(): score for score, group in score_map.items() for n in group}

# Streamlit UI setup
st.set_page_config(page_title="Lagos Rent Estimator", layout="centered")
st.title("🏠 Lagos House Rent Price Estimator")

st.markdown("""
Use this app to predict the estimated rent price of a house in Lagos based on property details and location features.
""")

st.subheader("📋 Property Features")

# Convert Yes/No to 1/0
def binary_input(label, help_text=""):
    return 1 if st.selectbox(label, ['Yes', 'No'], help=help_text) == 'Yes' else 0

# Binary inputs as Yes/No dropdowns
serviced = binary_input("Is the property serviced?", "Includes services like electricity, water, cleaning, etc.")
newly_built = binary_input("Is it newly built?", "1 if the building is newly constructed.")
furnished = binary_input("Is it furnished?", "1 if the house includes furniture like beds, chairs, etc.")


bedrooms = st.number_input("Number of Bedrooms", min_value=0, max_value=10)
bathrooms = st.number_input("Number of Bathrooms", min_value=0, max_value=10)

st.subheader("🌍 Location Information")

# City dropdown
city = st.selectbox("Select the City", [
    'Lekki', 'Ikoyi', 'Ikeja', 'Island', 'Ojodu', 'Gbagada',
    'Surulere', 'Ajah', 'Yaba'
])

# Neighborhood dropdown
selected_neighborhood = st.selectbox("Select Neighborhood", neighborhood_list)

# Calculate area_score from neighborhood
area_score = neighborhood_lookup.get(selected_neighborhood, 2)

# Auto-calculate luxury features and location premium
luxury_count = serviced + newly_built + furnished
location_premium = area_score + (luxury_count * 0.5)

# Display derived features
#Do not display
#st.markdown(f"📈 **Area Score**: `{area_score}` (based on neighborhood)")
#st.markdown(f"✨ **Luxury Count**: `{luxury_count}` (Serviced + Newly Built + Furnished)")
#st.markdown(f"💎 **Location Premium**: `{location_premium}` (Auto-calculated)")

#st.markdown("---")

# Predict button
if st.button("🔍 Predict Rent Price"):
    input_data = {
        'Serviced': serviced,
        'Newly Built': newly_built,
        'Furnished': furnished,
        'Bedrooms': bedrooms,
        'Bathrooms': bathrooms,
        'City': city,
        'Area_Score': area_score,
        'Luxury_Count': luxury_count,
        'Location_Premium': location_premium
    }

    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)[0]

    st.success(f"🏷️ Estimated Rent Price: **₦{prediction:,.2f}**")
