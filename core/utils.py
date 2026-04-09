import pandas as pd
import requests
import streamlit as st

@st.cache_data(ttl=3600) # Cache for 1 hour so it only runs once
def get_token_map():
    # Using the faster Cloudfront URL if possible, or the standard one
    url = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
    
    try:
        # Adding a timeout of 10 seconds so it doesn't spin forever
        response = requests.get(url, timeout=10)
        data = response.json()
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # CRITICAL: Filter immediately to reduce size! 
        # We only want NSE Equity stocks (no F&O, no BSE for now)
        df = df[(df['exch_seg'] == 'NSE') & (df['symbol'].str.endswith('-EQ'))]
        
        # Create a clean display name: "SBIN-EQ" -> "SBIN"
        df['display_name'] = df['symbol'].str.replace('-EQ', '')
        
        return df.set_index('display_name')['token'].to_dict()
        
    except Exception as e:
        st.error(f"Token Load Error: {e}")
        # Fallback to a few common tokens if the internet is slow
        return {"SBIN": "3045", "RELIANCE": "2885", "INFY": "1522"}