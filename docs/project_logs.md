# Daily Logs
## September 2025
### Week 01
11-09-2025
Opzetten van Gitomgeving met development notebooks, frameworks en projectscope gedefiniëerd in README.md.

12-09-2025
Logging sample code gemaakt voor EDA en evaluations. Daarnaast een basis EDA gemaakt voor inzichten.

16-09-2025
README.md geüpdatet, expertconsult en taakverdeling gemaakt.

### Week 02
19-09-2025
Kanban aangemaakt voor taakverdeling in Gitlab , frameworks vergeleken, data preprocessing en begin gemaakt aan de code voor vectoriseren.

21-09-2025
Data preprocessing van beide job_description datasets zoals droppen van onbruikbare kolommen en missende waardes vervangen.

### Week 03
23-09-2025
In de vectorisatie code verandert naar ander SBERT model. Ook heb ik een template gemaakt voor een data preprocessing pipeline waarin iedereen hun contributie in kan toevoegen voor een gestroomlijnde opschoning van de dataset.

24-09-2025
Bespreken van project, projectmanagement zoals het bijwerken van Kanban en helpen met Git opzetten bij andere groepsleden.

25-09-2025
Eerste run van data preprocessing pipeline om te testen, daarna de lengte van de rows onderzocht om te zien of het gekozen embedding model geschikt is, of dat we een ander model of andere aanpak moeten gebruiken.

## Oktober 2025
### Week 04
03-10-2025
Gezamenlijke code van data preprocessing gefixt en project besproken.

04-10-2025
EDA gemaakt van resume_data, kolommen gevonden die onbruikbaar waren met aantekeningen voor verdere bespreking. Ook missing values vervangen in de overige kolommen. Verder nog wat refactoring van code en opschonen van dev folder.

09-10-2025
Overleg met de groep. Mijn column drops en missing values code in de gezamenlijke resume preprocessing toegevoegd en wat aanpassingen gedaan aan de rest van de data preprocessing. Extra column drops omdat deze onbruikbaar zijn voor cv.

## January 2026
26-01-2026
Streamlit UI gebouwd met drie functionaliteiten: Single Match (CV vs vacature vergelijken), Batch Analysis (meerdere CVs ranken), en Search Jobs (zoeken in pre-geïndexeerde vacature database).

28-01-2026
Project structuur refactoring. Preprocessing geïntegreerd in de frontend zodat user input consistent wordt verwerkt met de embedding pipeline. Model gestandaardiseerd naar `all-mpnet-base-v2` voor consistentie. Text utilities module aangemaakt (`src/utils/text_utils.py`) voor herbruikbare text cleaning. README geüpdatet met nieuwe features en usage instructies.