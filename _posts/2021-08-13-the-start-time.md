---
layout: post
title: Reason for Audio-Video Async
---

In this post, I will summarize the events that reported the audio-video async and how they were resolved.

### The problem arises

I cut 1 minute video of "Lightning Network" video for demo purpose. However this video had something unusual. The video started almost a minute later as compared to audio. However, it took some time to realize this fact and each video player had a different way to present its defect. Neither the start time was presented in output from ffprobe. Thus, it remained a matter of concern for a couple of weeks.

However, it was also seen that async did not happen for uncut fresh videos. Thus, we misleadingly concluded that cutting using ffmpeg has been the reason for async. 

A lag of 1 sec was observed to be constant throughout the video. 

### The solution

When I played the video in video player of drive, the first 1 sec was clearly visible as missing. I realized the connection between this 1 sec and 1 sec of lag. So, it was clear that something the video was starting late by 1 sec. 

After more searching, I found that doing a ffprobe from python's ffmpeg shows the start time, and again I found the start time of 1 sec there. 

The mystery was solved. However, I don't remember what was the command that cut the video with such start times. 