In this file, I have compiled the results of some weird experiments which may or may not be of use in future. All of these experiments were done on a mini dataset of 2 sequences of full images (v_boat and i_castle).

-----
### Question 1) Does any detector give multiple keypoints in the same pixel?
Well, we are about to find out. Below is an experiment over a mini dataset of 2 sequences.

**ORB:** *Yes*, experiments show that there are numerous keypoints within a distance of 1 pixel. Also note that during this experiment, I do not count the distance of a keypoint from itself.
```bash
==================== TASK 1 ====================
============= Spread of Less than 1 px =============
1 out of 12 i_castle 1.ppm
 Number of KeyPoints 500
 Number of Distances<1 152.0
 .
 .
 .
7 out of 12 v_boat 1.ppm
 Number of KeyPoints 500
 Number of Distances<1 185.0
 .
 .
 .
Average less than 1 px distances 177.30769230769232
```

**SIFT:** *Yes* again
```bash
============= Spread of Less than 1 px =============
1 out of 12 i_castle 1.ppm
 Number of KeyPoints 1630
 Number of Distances<1 402.0
 .
 .
 .
Average less than 1 px distances 728.4615384615385
```

**SFOP:** *Yes*, but the number is very low compared to others
```bash
============= Spread of Less than 1 px =============
1 out of 12 i_castle 1.ppm
 Number of KeyPoints 1047
 Number of Distances<1 7.0
 .
 .
 Average less than 1 px distances 5.230769230769231
```
**LIFT:** *Yes*, with low number again
```bash
============= Spread of Less than 1 px =============
1 out of 12 i_castle 1.ppm
 Number of KeyPoints 920
 Number of Distances<1 8.0
 .
 .
 .
 Average less than 1 px distances 5.615384615384615 
```

**SuperPoint:**
*No*, and I did not expect this!
```bash
============= Spread of Less than 1 px =============
1 out of 12 i_castle 1.ppm
 Number of KeyPoints 913
 Number of Distances<1 0.0
 .
 .
 .
 Average less than 1 px distances 0.0
```

**D2Net**
*Yes*, but the number is very very low
```bash
============= Spread of Less than 1 px =============
1 out of 12 i_castle 1.ppm
 Number of KeyPoints 2657
 Number of Distances<1 2.0
 .
 .
 Average less than 1 px distances 1.1538461538461537
```

-----
### Question 2) Does your OpenCV SIFT implementation double the number of size of the image?
Well, I did some research. The SIFT implementation that I used was in the `xfeatures2d` namespace. A quick look at the source code [here](https://github.com/opencv/opencv_contrib/blob/master/modules/xfeatures2d/src/sift.cpp) makes me draw following conclusions:  
* Image size is doubled if `firstOctave < 0`
* If `useProvidedKeypoints` (which is `False`, because we `detectAndCompute`), then `firstOctave = 0`
* Default value of `firstOctave = -1`
* In short if you use detect SIFT keypoints (and not use some list of kps you yourself chose using some heuristics), it will double the size of the image. 

-----
### Question 3) It's a Correct Match! But wait, how do you define a match? And what the hell do you mean when you say the match is Correct?
Okay, this is a hard one. Every paper defines its own definition of what it considers a match. 

**Meaning of Match:** Nearest neighbour of a query descriptor in the list of target descriptors.   
**Meaning of Correct:** Now that you have a match, use ground truth homography to get the reprojection error in the position of keypoints. Match is correct if this error is below a particular threshold. To choose this threshold, I did some experiments. *You can see how the NN reprojection error is distributed in the file* `*_best_match.png`.

**My Conclusion:** There is no particular value of reprojection error for which you can call the match 'correct'. The one thing you can do is to say: when correct means $\epsilon = 1$, these are the results; when correct means $\epsilon = 3$, these are the results and so on.

-----
### Question 4) I think that different methods work better for different features when it comes to homography estimation. Is it true?
Performed some experiments over the complete dataset.  
Let us compare the overall error in the Homography estimation with and without Lowe's ratio test (best/second_best < 0.7). Note that Fails means that `None` is returned by OpenCV find_Homography or there are not enough 'good' matches.  
**My Conclusion:** Homography estimation works much better without Lowe's ratio test

| Descriptor | Overall Error (with Lowe's RT) | Overall Error (w/o LRT) | 
| ---------- | -------------                  | --------------------    |
|ORB         |  817.533   [Fails = 140 pairs] |  284.504   [Fails = 0]  |
|SIFT        |  167.909          [Fails = 2]  |  144.964   [Fails = 0]  |
|LIFT        |  139.454         [Fails = 20]  |  81.964    [Fails = 0]  |
|SUPERPOINT  |  88.222          [Fails = 22]  |  25.710    [Fails = 0]  |
|D2NET       |  369.483         [Fails = 99]  |  65.360    [Fails = 0]  |


Now that we know that homography estimation is much better without the Lowe Ratio test, we perform homography estimation *using NN only*.  
See the percentage of pairs with correctly predicted homography. Correct means that the h_err < 5 px:

| Descriptor    | %age Correct (Higher == better)   |
| -----------   | -------------     |
| ORB           | 0.4431            |
| SIFT          | 0.6913            |
| LIFT          | 0.7344            |
| SUPERPOINT    | 0.8310            |
| D2NET         | 0.5327            |

------

### Question 5) But why did you decide the ratio of (best_match_dist/second_best_dist) to be 0.7 for Ratio Test?

I think it was not good to take this ratio for all the features. Hence, I performed some experiments on the entire dataset to see what ratio suits the best for which features. I wanted to create individual ratio tests for each of the features.

**Distance between descriptors = euclidean**  
**My Conclusion:** It is not fair to use ratio test for all features. For example, ratio test on ORB and D2Net remove almost 50% of the correct matches (reprojection error < 4 px) while removing 90% of the incorrect ones.

```bash
atripath@photolab84:~/Downloads/fancy-keypoints/results/experiments$ more orb_ratio_test.txt 
orb
Remove at least 85.0 percent of the incorrect matches 
 Percentage of incorrect matches removed at 0.85: 0.9225364838836219 
 Percentage of correct matches removed at 0.85: 0.4974160823037415 
      Percentage of incr mat removed at 0.95: 0.5604568033462476 
      Percentage of corr mat removed at 0.95: 0.19998727917441841
---------------------------------------------------------------------
atripath@photolab84:~/Downloads/fancy-keypoints/results/experiments$ more sift_ratio_test.txt 
sift
Remove at least 85.0 percent of the incorrect matches 
 Percentage of incorrect matches removed at 0.85: 0.9136447857449231 
 Percentage of correct matches removed at 0.85: 0.16157081892372643 
      Percentage of incr mat removed at 0.95: 0.582868302963929 
      Percentage of corr mat removed at 0.95: 0.05316629640656849 
---------------------------------------------------------------------
atripath@photolab84:~/Downloads/fancy-keypoints/results/experiments$ more lift_ratio_test.txt 
lift
Remove at least 85.0 percent of the incorrect matches 
 Percentage of incorrect matches removed at 0.85: 0.8810889531940697 
 Percentage of correct matches removed at 0.85: 0.17525960905929894 
      Percentage of incr mat removed at 0.95: 0.5172663019498185 
      Percentage of corr mat removed at 0.95: 0.056550136265388594 
---------------------------------------------------------------------
atripath@photolab84:~/Downloads/fancy-keypoints/results/experiments$ more superpoint_ratio_test.txt 
superpoint
Remove at least 85.0 percent of the incorrect matches 
 Percentage of incorrect matches removed at 0.85: 0.8744840903205593 
 Percentage of correct matches removed at 0.85: 0.1916336053890639 
      Percentage of incr mat removed at 0.95: 0.5632247112344427 
      Percentage of corr mat removed at 0.95: 0.05954661413176697 
---------------------------------------------------------------------
atripath@photolab84:~/Downloads/fancy-keypoints/results/experiments$ more d2net_ratio_test.txt 
d2net
Remove at least 85.0 percent of the incorrect matches 
 Percentage of incorrect matches removed at 0.9: 0.8686902575937435 
 Percentage of correct matches removed at 0.9: 0.48955769732034204 
      Percentage of incr mat removed at 0.95: 0.7169318701434704 
      Percentage of corr mat removed at 0.95: 0.2938532736538944 
----------------------------------------------------------------------

```