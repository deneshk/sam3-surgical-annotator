# For Implementation


# Backburner tasks for review later

- Need to take numbers off of box labels
- Need to adjust box labels if theyre in the corner of the screen 
- If I try to redraw the prompt, be able to undo so that the old segementation is recovered 
- Ensure multiple of the same class can be saved 
- Better visibility and aesthetics on the resize ui for the boxes
- Resize predicted boxes option
- Method to save progress and resume where it was left off 
- When revising propagated frames, consider rerunning the segmentation with the current propagated mask + the new revision. 


- Consider "Propagate on next frame
- Consider removing segmentations and just having boxes
- Consider removing prompt boxes and just having boxes 
	- Need to clarify that first segment 
- Review SAM3 feature extracted frames for possible patterns

- Can consider translating the point prompts along with the box prompt

# Learnings

- Consider the use of back propagation in order to select better intial frame to initiate tracking