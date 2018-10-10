---
title: Boursier, Collecteur de notes boursières sur le site Boursier.com
author: thomas
date: '2018-10-10'
slug: []
categories: []
tags: []
header:
  caption: ''
  image: ''


---



# README

## Boursier

Boursier est une librairie conçue pour collecter des notes sur le site https://www.boursier.com/, dans l'onglet actualité. La collecte de note se fait grâce à des mots clés inscrits dans une liste de valeur Python. La collecte de note est ensuite sauvegardé en format Word. Le programme peut être lancé à différent moment de la journée, cela ne posera pas de problème avec les notes sauvegardées auparavant. En fin de journée, l'ensemble des notes peuvent être consolidées dans un seul et même fichier Word.

## Objectif principale

L'idée derrière ce programme est d'automatisée le processus de collecte de note depuis le site d'information boursière https://www.boursier.com/.

![](https://github.com/thomaspernet/scrap_boursier/raw/master/image/1.jpeg)



L'utilisateur peut collecter des notes sur des valeurs boursières à différent moment de la journée, et en fin de journée consolider l'ensemble des notes, modifiées au préalable ou non, dans un seul fichier word. 

Prenons l'exemple suivant, l'utilisateur décide de collecter les valeurs A et B le matin. Dans ce cas là, le programme va rechercher toutes les notes de ses deux valeurs et les enregistrer en format Word. L'utilisateur décide de modifier les notes, cela est possible sans poser problème à la consolidation des données en fin de journée. 

En milieu d'après-midi, l'utilisateur souhaite actualiser les notes depuis la dernière collecte. Le programme va automatiquement mettre à jour les notes depuis la dernière collecte. 

En fin de journée, le programme va consolider toutes les notes qui ont été recherchées durant la journée. 

## Guide d'utilisation

Le programme est conçu avec trois fonctions principales et une fonction sous-jatente. 

Attention, il y a d'autres fonctions sous-jacentes utilisées durant la collecte, toutefois, elles n'ont pas d'intérêt direct pour l'utilisateur. 

Le processus se divise en trois étapes:

1. Définition des paramètres et création des fichiers Word
2. Mise à jour des notes colléctées
3. Consolidation des notes

![](https://github.com/thomaspernet/scrap_boursier/raw/master/image/2.png)

### Définition des paramètres

L'utilisateur doit renseigner quatre paramètres:

- L'URL
- horaire de début
- horaire de fin
- valeur

**URL**

Sur le site https://www.boursier.com/., les notes sont quotidiennes sont situées à l'adresse suivante:

- https://www.boursier.com/actualites/news-du-jour

Pour les notes des jours précédents, il suffit de rajouter la date au format ANNEEJOURMOIS pour le jour souhaité. Example, Mardi 10 septembre : 20181009, l'adresse devient:

- https://www.boursier.com/actualites/news-du-jour/20181009

**Horaire de début**

L'horaire de début désigne la première borne de l'interval de temps à collecter. Par exemple, le début de la collecter peut commencer à minuit du même jour, la valeur de début devient:

- `datetime.datetime.strptime('2018-10-09T00:00:00', "%Y-%m-%dT%H:%M:%S")`

Notez qu'il est nécéssaire de modifier également la date. 

**Horaire de fin**

L'horaire de fin définit la seconde borne de l'interval de temps, c'est à dire, lorsque la collecte doit s'arrêter. Par exemple, la collecte peut être réalisée jusqu'à treize heures.

- `datetime.datetime.strptime('2018-10-09T13:00:00', "%Y-%m-%dT%H:%M:%S")`

**Valeur**

Les valeurs à collecter doivent être inscrites dans une liste python. Pour la collecte des valeurs  `wall-street` et `renault`, il faut noter:

- `["wall-street", "renautl"]`

Notez qu'il n'y a jamais de majuscule et un tiret si la valeur est composée de deux mots. En cas de doute, veuillez vous reportez à une adresse URL contenant l'un des mots clés

Example:

![](https://github.com/thomaspernet/scrap_boursier/raw/master/image/3.png)

### Création fichier Word

Le programme va créer un fichier Word distinct pour **toutes** nouvelles valeurs collectées et ensuite actualisées le fichier relatif aux valeurs (étapes deux). Il est donc nécéssaire de *paramètrer la classe à chaque fois que l'on modifie les paramètres*

*Paramètre*s

```python
### Importer les librairies necessaires
import datetime
import time
#import scrap_title as bt
import Boursier as br
###

url = "https://www.boursier.com/actualites/news-du-jour/20181009"
horaire_debut = datetime.datetime.strptime('2018-10-09T00:00:00', "%Y-%m-%dT%H:%M:%S")
horaire_fin = datetime.datetime.strptime('2018-10-009T13:00:00', "%Y-%m-%dT%H:%M:%S")
value =["wall-street","renault"]
```

Pour paramètrer le programme, il faut utiliser la classe `extract_content_Boursier`. 

```python
Scrap = br.extract_content_Boursier(url = url, horaire_debut = horaire_debut, horaire_fin = horaire_fin)

```

Pour créer les fichiers Word, il faut utiliser la fonction `write_word()` dans une `list_comprehension`. La fonction `write_word` a seulement un seul argument, la valeur à collecter

```python
### Create word files
[Scrap.write_word(value = x) for x in value]
```

Pour connaitre les URL collecter pour une valeur, il est possible d'utiliser la fonction `get_url()` qui comporte un seul argument, la valeur recherchée

```python
Scrap.get_url("wall-street")
```

Chaque fichier Word va être nommer de la façon suivante:

- Nom de la valeur +  heure à laquelle le programme est lancé

Il est possible de faire une modification des fichiers Word. Toutefois, vous devez toujours sauvegarder le fichier sous le même nom.

Notez que si vous souhaitez ajouter des nouvelles valeurs, vous devez toujours utiliser la fonction `create_word()` avant de mettre à jour des valeurs. 

## Mise à jour fichiers

La mise à jour des fichiers est relativement simple, il faut changer les plages horaires à  collecter, modifier les paramètres de `extract_content_Boursier()`et utiliser la fonction `update_word()`

```python
### Changement des plages et horaire
horaire_debut = datetime.datetime.strptime('2018-10-05T13:00:01', "%Y-%m-%dT%H:%M:%S")
horaire_fin = datetime.datetime.strptime('2018-10-05T23:00:00', "%Y-%m-%dT%H:%M:%S")
Scrap = br.extract_content_Boursier(url = url, horaire_debut = horaire_debut, horaire_fin = horaire_fin)

### Mise à jour fichier word
[Scrap.update_word(value = x) for x in value]
```

Les fichiers Word sont sauvegarder de la même manière que `write_word()`, valeur+heure. 

Si vous souhaitez modifier un fichier Word après avoir mis a jour, vous devez toujours utiliser le document Word le plus récent. 

Veuillez faire attention à avoir créer le fichier Word pour toutes les valeurs à collecter avant de mettre à jour. 

### Consolidation

La consolidation se fait avec la fonction `wd_md_wd()`. La fonction comporte un seul argument, la liste des valeurs. 

```python
### Consolide fichier word
Scrap.wd_md_wd(value = value, name_final = "wall_street_renault")
```

En lançant cette ligne, le programme va consolider toutes les documents Word présent dans le dossier et le nommer `wall_street_renalt.docx`.

Dans cette étape, vous pouvez choisir quelles valeurs à  consolider. Par exemple, vous avez collecter les valeurs A, B, C et D et vous souhaitez consolider A et B ensemble puis C et D alors vous devez changer la liste de valeur. 

```python
### A et B
valeur_1 = ["A", "B"]
Scrap.wd_md_wd(value = valeur_1, name_final = "A_B")

### C et D
valeur_2 = ["C", "D"]
Scrap.wd_md_wd(value = valeur_2, name_final = "C_D")
```

