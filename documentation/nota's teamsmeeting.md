**route** is afhankelijk van dag waarop ze gereden wordt, omgeving waar gereden wordt en het depot id
2 zelfde route realisaties zijn niet altijd ehtzelfde: 
route id en depot id is NIET uniek!! op verschillende dagen kan één route gereden worden. 
- meerdere vrachtwagens: één depot heeft meerdere routes. 
- één unieke route die gereden wordt is DATUM + route_id


- hoe bepalen wanneer op één dag een route ten einde komt en wanneer er een tweede route gepland wordt?
    - je hebt 's ochtends een paar requests.
    - de avond request 'createSequence' negeren
    - de laatste op de dag met 'estimateTime' (kan ook een andere zijn) wordt dan effectief gereden
        - estimateTime is een slechte - ga gewoon uit van de allerlaatste route die gepland wordt. 

    - pipeline: 
        - eerste request van de dag vergelijken met de laatste request van de dag dat niet in de avond valt!


# OPTIES: 
- OR algo(PPT) is computationeal heavy - nood aan optimalisatie ligt daar. 
    - TAAK: hoe kan het proces verbeterd worden met ML
    - Bedenk een ML-oplossing die locaties kan uitfilteren voor het naar de OR gaat (zodat je niet steeds de hele route moet berekenen)

- Huidig OR algo kan je ook optimaliseren 

- Of response onderscheppen en analyseren. 


# DELIVERABLES:
- stel je hebt een goed model:
    1) Welk deel van de operaties is dan geïmpacteerd en waar gebeurt de implementatie.
    2) Hoe ga je de data manipuleren tot iets waar je je model op kan trainen
    3) Voer een evaluatie uit voor je model; kan je de doelstelling bereiken? Maak de analyse over het al dan niet missen van het doel. 

Ideetje voor toevoegen: 
- plot alles en zoek de nearest neighbour van een punt OBV LAT LON van een punt dat in éénzelfde dag gereden wordt. 

Herordenen van routes: 
- Coordinaten zijn niet echt, werkt dus niet door gebruik te maken van Open Street Map en NEO4J
- deel van de denkoefening kan zijn: Stél dat je de ehcte data hebt, hoe zou het dan helpen bij een betere performance van de oplossing?
- Routes downbreaken in kleinere routes KAN: er zijn theoretisch meerdere cammions beschikbaar. (dit is een advanced issue.)


# CONCREET: 
1) je wilt een functie die op één route met alle gekende punten:
    - de kans berekent dat een put in die route verwijderd wordt. 
    e.g. ```
        def chanceOfPointRemoval(route, point):
            implement
        
        ```

    - Kan misschien ook met hexagonen: <-> toch eerder gewoon clustering gebruiken.
        * plot alle data op een 'map', doe een overlay van hexagonen.
        * bereken dan hoeveel punten in één hexagoon zit
        * bereken dan hoeveel punten in de nabije hexagonen zitten. zoiets: https://www.flerlagetwins.com/2018/11/what-hex-brief-history-of-hex_68.html 
        * techniek heet: hexagonal binning (https://gisgeography.com/hexagon-binning/) 
        * door die clustering kan je centra herkennen. 
        denk dan na wat is de afstand van een punt tot dichtste centrum. 
        

2) Mee vermelden in de eindopdracht dat rechte lijnen niet kunnen capteren hoe ver je effectief moet omrijden (denk aan rivieren met een brug waar je 20 KM voor moet omrijden)
    - zoek naar k-dichtsbijzijnde punten
    - Kan je doen voordat de OR gedaan wordt.
    - Probeer proactief te voorspellen of een route te lang gaat zijn (voor het OR algo.)
        ==> model dat by proxy de afgelegde afstand voorspelt OBV/ Euclidische afstand.
        - Als je een route hebt die te lang is, dan kun je proactief een route flaggen. (stuur je dan nog niet naar de OR)



3) Tijdstippen kunenn handig zijn (totale reistijd) - maar je kan er een proxy voor nodig hebben
- tijd kan je vervangen door gereden afstand tot op een bepaald punt. 
    =+> let op: Afstanden zijn geschaald!! lijken héél klein. 

- Tijdstip en afgelegde afstand KAN JE PAS berekenen NADAT er een route berekend is door de OR algo



4) trainingsdata voor routes toe te voegen:
- Kan je artificieel maken door uit één punt de effectief gereden routes te verwijderen
- dan heb je trainingsdata
- daar kan je op trainen
- een algo is goed ALS je het punt dat er uit is gehaald correct wordt toegevoegd.
- voeg dan ook de punten toe die historisch verwijderd zijn; dan kan je je model doen trainen op data met punten die wél en niet goed zijn :)




