# Importing libraries
import numpy as np
import streamlit as st
import pickle
import sklearn

# Loading the model to predict on the data
pickle_in = open('linear', 'rb')
linear = pickle.load(pickle_in)

columns = np.array(['bhk', 'total_sqft', 'bath', '1st Block Jayanagar',
       '1st Phase JP Nagar', '2nd Phase Judicial Layout',
       '2nd Stage Nagarbhavi', '5th Block Hbr Layout',
       '5th Phase JP Nagar', '6th Phase JP Nagar', '7th Phase JP Nagar',
       '8th Phase JP Nagar', '9th Phase JP Nagar', 'AECS Layout',
       'Abbigere', 'Akshaya Nagar', 'Ambalipura', 'Ambedkar Nagar',
       'Amruthahalli', 'Anandapura', 'Ananth Nagar', 'Anekal',
       'Anjanapura', 'Ardendale', 'Arekere', 'Attibele', 'BEML Layout',
       'BTM 2nd Stage', 'BTM Layout', 'Babusapalaya', 'Badavala Nagar',
       'Balagere', 'Banashankari', 'Banashankari Stage II',
       'Banashankari Stage III', 'Banashankari Stage V',
       'Banashankari Stage VI', 'Banaswadi', 'Banjara Layout',
       'Bannerghatta', 'Bannerghatta Road', 'Basavangudi',
       'Basaveshwara Nagar', 'Battarahalli', 'Begur', 'Begur Road',
       'Bellandur', 'Benson Town', 'Bharathi Nagar', 'Bhoganhalli',
       'Billekahalli', 'Binny Pete', 'Bisuvanahalli', 'Bommanahalli',
       'Bommasandra', 'Bommasandra Industrial Area', 'Bommenahalli',
       'Brookefield', 'Budigere', 'CV Raman Nagar', 'Chamrajpet',
       'Chandapura', 'Channasandra', 'Chikka Tirupathi', 'Chikkabanavar',
       'Chikkalasandra', 'Choodasandra', 'Cooke Town', 'Cox Town',
       'Cunningham Road', 'Dasanapura', 'Dasarahalli', 'Devanahalli',
       'Devarachikkanahalli', 'Dodda Nekkundi', 'Doddaballapur',
       'Doddakallasandra', 'Doddathoguru', 'Domlur', 'Dommasandra',
       'EPIP Zone', 'Electronic City', 'Electronic City Phase II',
       'Electronics City Phase 1', 'Frazer Town', 'GM Palaya',
       'Garudachar Palya', 'Giri Nagar', 'Gollarapalya Hosahalli',
       'Gottigere', 'Green Glen Layout', 'Gubbalala', 'Gunjur',
       'HAL 2nd Stage', 'HBR Layout', 'HRBR Layout', 'HSR Layout',
       'Haralur Road', 'Harlur', 'Hebbal', 'Hebbal Kempapura',
       'Hegde Nagar', 'Hennur', 'Hennur Road', 'Hoodi', 'Horamavu Agara',
       'Horamavu Banaswadi', 'Hormavu', 'Hosa Road', 'Hosakerehalli',
       'Hoskote', 'Hosur Road', 'Hulimavu', 'ISRO Layout', 'ITPL',
       'Iblur Village', 'Indira Nagar', 'JP Nagar', 'Jakkur', 'Jalahalli',
       'Jalahalli East', 'Jigani', 'Judicial Layout', 'KR Puram',
       'Kadubeesanahalli', 'Kadugodi', 'Kaggadasapura', 'Kaggalipura',
       'Kaikondrahalli', 'Kalena Agrahara', 'Kalyan nagar', 'Kambipura',
       'Kammanahalli', 'Kammasandra', 'Kanakapura', 'Kanakpura Road',
       'Kannamangala', 'Karuna Nagar', 'Kasavanhalli', 'Kasturi Nagar',
       'Kathriguppe', 'Kaval Byrasandra', 'Kenchenahalli', 'Kengeri',
       'Kengeri Satellite Town', 'Kereguddadahalli', 'Kodichikkanahalli',
       'Kodigehaali', 'Kodigehalli', 'Kodihalli', 'Kogilu', 'Konanakunte',
       'Koramangala', 'Kothannur', 'Kothanur', 'Kudlu', 'Kudlu Gate',
       'Kumaraswami Layout', 'Kundalahalli', 'LB Shastri Nagar',
       'Laggere', 'Lakshminarayana Pura', 'Lingadheeranahalli',
       'Magadi Road', 'Mahadevpura', 'Mahalakshmi Layout', 'Mallasandra',
       'Malleshpalya', 'Malleshwaram', 'Marathahalli', 'Margondanahalli',
       'Marsur', 'Mico Layout', 'Munnekollal', 'Murugeshpalya',
       'Mysore Road', 'NGR Layout', 'NRI Layout', 'Nagarbhavi',
       'Nagasandra', 'Nagavara', 'Nagavarapalya', 'Narayanapura',
       'Neeladri Nagar', 'Nehru Nagar', 'OMBR Layout', 'Old Airport Road',
       'Old Madras Road', 'Padmanabhanagar', 'Pai Layout', 'Panathur',
       'Parappana Agrahara', 'Pattandur Agrahara', 'Poorna Pragna Layout',
       'Prithvi Layout', 'R.T. Nagar', 'Rachenahalli',
       'Raja Rajeshwari Nagar', 'Rajaji Nagar', 'Rajiv Nagar',
       'Ramagondanahalli', 'Ramamurthy Nagar', 'Rayasandra',
       'Sahakara Nagar', 'Sanjay nagar', 'Sarakki Nagar', 'Sarjapur',
       'Sarjapur  Road', 'Sarjapura - Attibele Road',
       'Sector 2 HSR Layout', 'Sector 7 HSR Layout', 'Seegehalli',
       'Shampura', 'Shivaji Nagar', 'Singasandra', 'Somasundara Palya',
       'Sompura', 'Sonnenahalli', 'Subramanyapura', 'Sultan Palaya',
       'TC Palaya', 'Talaghattapura', 'Thanisandra', 'Thigalarapalya',
       'Thubarahalli', 'Tindlu', 'Tumkur Road', 'Ulsoor', 'Uttarahalli',
       'Varthur', 'Varthur Road', 'Vasanthapura', 'Vidyaranyapura',
       'Vijayanagar', 'Vishveshwarya Layout', 'Vishwapriya Layout',
       'Vittasandra', 'Whitefield', 'Yelachenahalli', 'Yelahanka',
       'Yelahanka New Town', 'Yelenahalli', 'Yeshwanthpur'])

# Defining the function which will make the prediction
def predict_price(location, bhk, sqft, bath):

    loc_index = np.where(columns == location)[0][0]  # It gives the index of the 'location' column specifically

    z = np.zeros(len(columns), dtype=int)  # Return a new array of given shape and size, filled with zeros.
    z[0] = bhk
    z[1] = sqft
    z[2] = bath
    if loc_index >= 3:
        z[loc_index] = 1

    price = linear.predict([z])[0] * 100000
    return round(price)

# Designing web page
st.write('##### Enter the details below to get your _Dream House Price_ predicted...')
st.sidebar.markdown('## Prediction')

location = st.selectbox('**Location**', ['1st Block Jayanagar', '1st Phase JP Nagar', '2nd Phase Judicial Layout', '2nd Stage Nagarbhavi',
       '5th Block Hbr Layout', '5th Phase JP Nagar', '6th Phase JP Nagar',
       '7th Phase JP Nagar', '8th Phase JP Nagar', '9th Phase JP Nagar',
       'AECS Layout', 'Abbigere', 'Akshaya Nagar', 'Ambalipura',
       'Ambedkar Nagar', 'Amruthahalli', 'Anandapura', 'Ananth Nagar',
       'Anekal', 'Anjanapura', 'Ardendale', 'Arekere', 'Attibele',
       'BEML Layout', 'BTM 2nd Stage', 'BTM Layout', 'Babusapalaya',
       'Badavala Nagar', 'Balagere', 'Banashankari',
       'Banashankari Stage II', 'Banashankari Stage III',
       'Banashankari Stage V', 'Banashankari Stage VI', 'Banaswadi',
       'Banjara Layout', 'Bannerghatta', 'Bannerghatta Road',
       'Basavangudi', 'Basaveshwara Nagar', 'Battarahalli', 'Begur',
       'Begur Road', 'Bellandur', 'Benson Town', 'Bharathi Nagar',
       'Bhoganhalli', 'Billekahalli', 'Binny Pete', 'Bisuvanahalli',
       'Bommanahalli', 'Bommasandra', 'Bommasandra Industrial Area',
       'Bommenahalli', 'Brookefield', 'Budigere', 'CV Raman Nagar',
       'Chamrajpet', 'Chandapura', 'Channasandra', 'Chikka Tirupathi',
       'Chikkabanavar', 'Chikkalasandra', 'Choodasandra', 'Cooke Town',
       'Cox Town', 'Cunningham Road', 'Dasanapura', 'Dasarahalli',
       'Devanahalli', 'Devarachikkanahalli', 'Dodda Nekkundi',
       'Doddaballapur', 'Doddakallasandra', 'Doddathoguru', 'Domlur',
       'Dommasandra', 'EPIP Zone', 'Electronic City',
       'Electronic City Phase II', 'Electronics City Phase 1',
       'Frazer Town', 'GM Palaya', 'Garudachar Palya', 'Giri Nagar',
       'Gollarapalya Hosahalli', 'Gottigere', 'Green Glen Layout',
       'Gubbalala', 'Gunjur', 'HAL 2nd Stage', 'HBR Layout',
       'HRBR Layout', 'HSR Layout', 'Haralur Road', 'Harlur', 'Hebbal',
       'Hebbal Kempapura', 'Hegde Nagar', 'Hennur', 'Hennur Road',
       'Hoodi', 'Horamavu Agara', 'Horamavu Banaswadi', 'Hormavu',
       'Hosa Road', 'Hosakerehalli', 'Hoskote', 'Hosur Road', 'Hulimavu',
       'ISRO Layout', 'ITPL', 'Iblur Village', 'Indira Nagar', 'JP Nagar',
       'Jakkur', 'Jalahalli', 'Jalahalli East', 'Jigani',
       'Judicial Layout', 'KR Puram', 'Kadubeesanahalli', 'Kadugodi',
       'Kaggadasapura', 'Kaggalipura', 'Kaikondrahalli',
       'Kalena Agrahara', 'Kalyan nagar', 'Kambipura', 'Kammanahalli',
       'Kammasandra', 'Kanakapura', 'Kanakpura Road', 'Kannamangala',
       'Karuna Nagar', 'Kasavanhalli', 'Kasturi Nagar', 'Kathriguppe',
       'Kaval Byrasandra', 'Kenchenahalli', 'Kengeri',
       'Kengeri Satellite Town', 'Kereguddadahalli', 'Kodichikkanahalli',
       'Kodigehaali', 'Kodigehalli', 'Kodihalli', 'Kogilu', 'Konanakunte',
       'Koramangala', 'Kothannur', 'Kothanur', 'Kudlu', 'Kudlu Gate',
       'Kumaraswami Layout', 'Kundalahalli', 'LB Shastri Nagar',
       'Laggere', 'Lakshminarayana Pura', 'Lingadheeranahalli',
       'Magadi Road', 'Mahadevpura', 'Mahalakshmi Layout', 'Mallasandra',
       'Malleshpalya', 'Malleshwaram', 'Marathahalli', 'Margondanahalli',
       'Marsur', 'Mico Layout', 'Munnekollal', 'Murugeshpalya',
       'Mysore Road', 'NGR Layout', 'NRI Layout', 'Nagarbhavi',
       'Nagasandra', 'Nagavara', 'Nagavarapalya', 'Narayanapura',
       'Neeladri Nagar', 'Nehru Nagar', 'OMBR Layout', 'Old Airport Road',
       'Old Madras Road', 'Other', 'Padmanabhanagar', 'Pai Layout',
       'Panathur', 'Parappana Agrahara', 'Pattandur Agrahara',
       'Poorna Pragna Layout', 'Prithvi Layout', 'R.T. Nagar',
       'Rachenahalli', 'Raja Rajeshwari Nagar', 'Rajaji Nagar',
       'Rajiv Nagar', 'Ramagondanahalli', 'Ramamurthy Nagar',
       'Rayasandra', 'Sahakara Nagar', 'Sanjay nagar', 'Sarakki Nagar',
       'Sarjapur', 'Sarjapur  Road', 'Sarjapura - Attibele Road',
       'Sector 2 HSR Layout', 'Sector 7 HSR Layout', 'Seegehalli',
       'Shampura', 'Shivaji Nagar', 'Singasandra', 'Somasundara Palya',
       'Sompura', 'Sonnenahalli', 'Subramanyapura', 'Sultan Palaya',
       'TC Palaya', 'Talaghattapura', 'Thanisandra', 'Thigalarapalya',
       'Thubarahalli', 'Tindlu', 'Tumkur Road', 'Ulsoor', 'Uttarahalli',
       'Varthur', 'Varthur Road', 'Vasanthapura', 'Vidyaranyapura',
       'Vijayanagar', 'Vishveshwarya Layout', 'Vishwapriya Layout',
       'Vittasandra', 'Whitefield', 'Yelachenahalli', 'Yelahanka',
       'Yelahanka New Town', 'Yelenahalli', 'Yeshwanthpur'])

bhk = st.slider('**BHK**', 1, 20, 1)

sqft = st.slider('**Total Area**', 300.0, 30000.0, 300.0)

bath = st.slider('**Bathroom**', 1, 20, 1)

if st.button('**Predict Price**'):
    price = predict_price(location, bhk, sqft, bath)
    st.success(f'**The predicted price of the house is Rs. {price}**')