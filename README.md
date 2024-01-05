**For fast whisper use version 0.7 

**Version 0.7 has fixed the break after the latest Google Colab upgrade. It has the same functionality as version 06i.



(https://colab.research.google.com/github/meizhong986/WhisperJAV/blob/main/notebook/WhisperJAV_v0_7.ipynb)


      
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/meizhong986/WhisperJAV/blob/main/notebook/WhisperJAV_v0_7.ipynb)
  
      




Note 1: hallucination removal is more robust in Japanese transcription than English translation task.

Note 2: it works mych faster with wav audio format.





**Welcome to WhisperJAV!**


This was put together for JAV enthusiasts that want to create subtitles for their favourite videos. The main goal for WhipserJAV is to be easy to use and fast for non-technical people. It uses faster-whisper to gain 2x speed of the original Whisper, and performs post processing to remove hallucination and repetition.

Please see the release section for versions of the WhisperJAV. You can open the latest version by clicking here Google Colab

For credits and citation please see the end of this document.



================================================

This guide provides step-by-step instructions on how to use WhisperJAV for your audio processing needs.



**1. Extract MP3 Audio from Videos** :

Several tools have been suggested in this community already. Some are: Clever FFmpeg, VLC Media Player, Audacity, or ffmpeg to extract audio. VLC and Audacity are user-friendly.

Here is Clever FFmpeg: [https://www.videohelp.com/software/clever-FFmpeg-GUI](https://www.videohelp.com/software/clever-FFmpeg-GUI)

Here's a tutorial for VLC **:** [https://youtu.be/sMy-T8RJAo0?si=AKg-WgDAAhtaBFkr](https://youtu.be/sMy-T8RJAo0?si=AKg-WgDAAhtaBFkr)  

  
**2. Create WhipserJAV Folder in Google Drive and upload mp3 there** :

To create a folder in Google Drive, click on the "+ New" button and select "Folder". Name it "WhisperJAV".

To upload files, open the "WhisperJAV" folder, click on "+ New" again, and select "File upload". Find your MP3 files and start the upload. You can also drag and drop.

 Here's couple of basic tutorials:

[https://youtu.be/EKjnjySLTvM?si=SF8ww3z572FnO\_cq](https://youtu.be/EKjnjySLTvM?si=SF8ww3z572FnO_cq)

[Organize your files in Google Drive - Computer - Google Drive Help](https://support.google.com/drive/answer/2375091?hl=en&co=GENIE.Platform%3DDesktop)  

**3. Run WhisperJAV:**

Open the Whisper template: (latest is version 0.6)

[https://colab.research.google.com/drive/1LTis2JcC66flJawHApGh29ZvKj4MuZl6#scrollTo=7zITdDF9fWEV](https://colab.research.google.com/drive/1LTis2JcC66flJawHApGh29ZvKj4MuZl6#scrollTo=7zITdDF9fWEV)

Run the template by clicking on top menu Runtime | Run all.

The template is a notebook in Google Colab. In Google Colab, the menu items are located at the top of the page: **File** : , **Edit** : , **View** : , **Insert** : , **Runtime** , **Tools**.

You want **Runtime**. This is the menu where you can run the template. For WhipserJAV you don't need any other menu item than 3 items under Runtime:

To run all cells in your notebook, you can go to Runtime \> Run all.

If you've made changes to options, you want to start fresh: choose Runtime \> Restart and run all. This will restart the runtime and then run all cells in order.

After you're done you want to Disconnect and delete.

If you receive any error message that doesn't make sense, usually those are caused by shared servers in Google. Just do a  Runtime \> Restart and run all.

Also pay attention that Google every now and then asks that you're not a robot 😊


**4. Accept Permission Request from Google Colab to Connect with Google Drive** :  

 When connecting the template with Google Drive, you'll be prompted to authorize access.
Upon prompt select you google account and accept to connect (authorise) at the end of the pop-up page. 




**5. Download Subtitles from Google Drive** :  

The subtitles are saved in WhisperJAV in your Google Drive by default. If you have changed the folder then look there. The subs are downloaded automatically as well as zipped and downloaded when runtime finishes. You can also download them one by one from your Drive.    

  
**==================================


**Credits and citation**  

Credits: @Anon\_entity, @phineas-pta, JAV communities Scanlover and Akiba

Upcoming features: More post-processing for accurate subs

For any comments, feature requests reach me at [meizhong.986@gmail.com](mailto:meizhong.986@gmail.com) or in the scanlover forum

**Version History:**

0.3: 1st public release

0.4: Changed UI and default folder

0.5: Added SRT post processing. Cleaned SRTs are in .cleaned.subs subfolder
