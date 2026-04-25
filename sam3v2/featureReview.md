# For Implementation


- turn off point prompt propagation

# Implemented features To Test/ Review


- Be able to get focus of a box prompt by double clicking and it will autoswitch to box mode 
- Add a way to flag specific frames to make it easy to jump to them in the timeline. 

# Backburner tasks for review later

- Need to take numbers off of box labels
- Need to adjust box labels if theyre in the corner of the screen 

- Ensure multiple of the same class can be saved 
- Better visibility and aesthetics on the resize ui for the boxes

- Consider removing prompt boxes and just having boxes 
	- Need to clarify that first segment 

- I think there's some lack of clarity on the prompt interface once there's some prompts on the screen 

- Can we improve 

- What if propagate updated the screen in real time, and then you could pause/stop if there's an error 


# Learnings and Other

- Consider the use of back propagation in order to select better intial frame to initiate tracking
- Review SAM3 feature extracted frames for possible patterns

- ONe possible reaosn point prompts aren't working super well is because the box prompt is overwriting it? 