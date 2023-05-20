# projet_compression L1 MI - TD3 


Auteurs du projet : LENNY BARBE et RIM GNICHI 



                    PROJET COMPRESSION JPEG



Les étapes clé de la compression :


- La Manipulation d'image 

Conversion des images RGB en YCBCr et inversement

- Le Traitement d'image 

Le padding : on va ajouter des pixels noir sur les lignes du bas et colonnes de droite jusqu'à ce que les dimensions de la matrice soient multiple de 8

L'anti-padding : retour à la matrice de depart sans remplissage du padding


- Le sous-echantillonage

Permet de rendre la matrice 2 fois plus petite en remplacent 2 pixel voisin de la matrice par la moyenne de ces 2 pixel

- L'anti sous-echantillonage

Permet de multiplier par 2 la dimension d'une image (dans notre exemple revenir à notre image de depart) 


- Le decoupage de la matrice

On decoupe la matrice pour qu'elle soit de dimension 8x8

- La transformée


        ...




- L'Option RLE

Methode de compression de donnée ici utilisé pour des éléments de matrice il s'agit de remplacer le nombre de 0 consecutif par seulement un seul 0 et #k (son nombre de répétition)



- La Decompression 


    ...




















