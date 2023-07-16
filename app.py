import streamlit as st
from recommender import restaurant_recommendations

def main():
    st.title("Restaurant Recommender System")

    # User inputs
    rest_name = st.text_input("Enter a restaurant name")
    rating_tolerance = st.slider("Rating Tolerance", 1.0, 5.0, step=0.1)

    # Button to trigger recommendation
    button_clicked = st.button("Get Recommendations")

    if button_clicked and rest_name:
        recommendations = restaurant_recommendations(rest_name, rating_tolerance)

        st.subheader("Top Recommendations:")
        for index, row in recommendations.iterrows():
            st.write("Restaurant Name:", row['rest_name'])
            st.write("Address:", row['address'])
            st.write("Average Rating:", row['rest_avg_stars'])
            st.write("Categories:", row['categories'])

if __name__ == '__main__':
    main()