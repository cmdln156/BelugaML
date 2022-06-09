import streamlit as st
from streamlit import caching
import pandas
import os.path
from os import path
import time
import matplotlib.pyplot as plt



#File selection function
def file_selector(folder_path='.'):
    filenames = os.listdir(folder_path)
    selected_filename = st.selectbox('Select a file:', filenames)
    return os.path.join(folder_path, selected_filename)



FPS=30 #Videos are 30 frames per second


INPUT_PATH="videos/"
SELECT_VIDEO_PATH="videos_to_select/"
filename=SELECT_VIDEO_PATH+"60ft_5x.mp4" #Set default selected filename.

#Hide menu and "Made By Streamlit" at bottom of page.
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

st.title('Automated Beluga and Boat Detection in UAV Imagery in the Churchill River Estuary')
st.header('Madison Harasyn, Wayne Chan, Emma Ausen, and David Barber')
st.subheader('Centre for Earth Observation Science, University of Manitoba')
st.set_option('deprecation.showfileUploaderEncoding', False)

st.text("")
st.write("The automated tracking of belugas in video is a challenging problem due to the whales being underwater most of the time. In typical tracking applications, such as the tracking of cars or pedestrians, the objects being tracked are in view most of the time and are only hidden (occluded) part of the time. It is the opposite with belugas: they are hidden most of the time and only visible (above water) a fraction of the time. In addition, belugas are difficult to distinguish from one another, particularly at a distance.")

st.write("Our study looks at the feasibility of using state-of-the-art machine learning algorithms to detect and track belugas in drone videos taken at an oblique angle. Videos were taken from a stationary position 60 feet above the ground on the southern shore of the Churchill River Estuary.")

st.write("Object detection was performed by Yolo v.4, a convolutional deep neural network model. The algorithm was trained to detect belugas, boats, and kayaks. Detections were passed to Deep SORT, a multiple object tracking algorithm.")

st.write("This demonstration allows you to experiment with a parameter of the tracking algorithm on two sample videos.")

st.text("")

#Select file.
filename = file_selector(SELECT_VIDEO_PATH)

st.text("")
st.write("The Deep SORT tracking algorithm uses a parameter called 'age', which is the length of time after which a tracked ID is considered to have left the field of view.") 
st.write("Try adjusting the age parameter to see its effect on the number of objects tracked in the video and the resulting statistics:")

age_slider = st.slider('Age (seconds)', 2, 30, 2, 2) #streamlit.slider(label, min_value=None, max_value=None, default value=None, step=None, format=None, key=None)

age_in_frames=age_slider*FPS


#base_outfilename="60ft_5x_age"+str(age_in_frames)+"_output"
basename=os.path.basename(filename)
base_outfilename=os.path.splitext(basename)[0]+"_age"+str(age_in_frames)+"_output"

video_filename=base_outfilename+".avi-converted.mp4"
text_filename=base_outfilename+"_summary.csv"

st.text("")
st.write("Detection and Tracking")

if (os.path.exists(INPUT_PATH+video_filename)):
    #st.write(INPUT_PATH+video_filename)
    st.video(INPUT_PATH+video_filename,format="video/mp4", start_time=0)
else: st.write('Video file not found.')

st.write("(Note: if the video doesn't load, try refreshing the browser.)")
st.text("")

if (path.exists(INPUT_PATH+text_filename)):
    df = pandas.read_csv(INPUT_PATH+text_filename,encoding='latin1',header=0)

    st.write('Objects Identified and Tracked')
    st.dataframe(df)

    st.write("Counts by Class")
    #Column 1 is the class and column 0 is the ID
    groups = df.groupby('Class')['ID'].nunique()
    st.write(groups)
    if basename == "60ft_9x.mp4":
        st.write("(Although it may seem that there were only four kayaks and not five as the count indicates, the edge of another kayak appears for a fraction of a second at around the 19 second mark of the video on the right side of the screen.)")
    
    st.text("")
    #Print descriptive statistics for beluga surfacing
    st.write("Summary Statistics for Visibility Time")
    #st.write(df[2].describe())
    st.write(df.groupby('Class')['Duration'].describe())

    beluga_times=df.query('Class == "beluga"')['Duration']
    
    st.text("")
    
    #Plot histogram
    fig,ax=plt.subplots()
    ax.hist(beluga_times,bins=10)
    ax.set_xlabel("Duration (Seconds)", labelpad=20, weight='bold', size=10)
    ax.set_ylabel("Frequency", labelpad=20, weight='bold', size=10)
    plt.title("Frequency of Beluga Surfacing Intervals")
    st.pyplot(fig)
else: st.write('Text file not found.')
