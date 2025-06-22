# advanced-dashboard-for-real-estate-analysis


This Streamlit app provides an advanced dashboard for real estate analysis and price prediction in King County, WA. It enables users to explore, visualize, and predict property prices using a pre-trained machine learning model, with features for geospatial analysis, data administration, and an integrated AI chatbot.

---

## Features

- **Interactive Dashboard:** View KPIs (average price, surface, price per sqft, etc.) and visualizations (scatter, box, histogram).
- **Advanced Analysis:** Correlation matrix, pairplots, and outlier detection for selected variables.
- **Predictive Modeling:** Real-time price estimation for custom property features using a Random Forest model.
- **Feature Importance:** Visualize the top factors influencing price predictions.
- **Geospatial Analysis:** Interactive map of property prices by zipcode, percentile coloring, and area statistics.
- **Data Administration:** Import/export datasets, handle missing values, and download predictions.
- **AI Chatbot:** Ask real estate questions and get instant answers powered by Groq's Llama 3 model.
- **Custom Styling:** Modern, responsive UI with custom CSS for enhanced user experience.

---

## Tech Stack

- Python
- Streamlit
- Pandas, NumPy
- Scikit-learn
- Plotly, Seaborn, Matplotlib, Folium
- Groq API (for chatbot)
- python-dotenv

---

## Getting Started

1. **Install dependencies:**

   ```sh
   pip install -r requirements.txt
