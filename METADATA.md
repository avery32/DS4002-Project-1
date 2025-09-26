**Data Summary:**  

**Provenance**  
This dataset stems from a research project into automated hate speech detection [1]. The initial dataset was a collection of tweets pulled from Twitter (now X). These tweets were pulled from a site called hatebase.org, which allows Twitter users to flag potentially hateful speech. 25,000 of those tweets were then used to create a new dataset, which utilized CrowdFlower to label them as hate speech, offensive language, or neutral. This new dataset included counts of how many individuals from Crowdflower viewed the tweet and their respective categorizations, as well as a final column called “count” that summarized the votes of those individuals by assigning a value of 2 for neutral, 1 for offensive speech, and 0 for hate speech.  
**License**  
This dataset is licensed under the MIT license, which allows it to be freely used by others and for projects like ours. This license further allows us to modify and merge the data for our purposes. 
**Ethical Statements:**
Please note that there is a layer of hate speech identification that is subjective. The line between offensive and hate speech is not always clearly drawn. To combat this, we used only data points where all Crowdflower users agreed on a classification for a tweet in order to help mitigate discrepencies. 
**Data Dictionary**  
![Data dictionary image](Data%20Dictionary.png)  
**EDA Plots**  

