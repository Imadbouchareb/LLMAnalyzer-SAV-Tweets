INTENT_LABELS = [
    "Facturation","Reseau/Internet","Mobile/SIM/Portabilite","Box/TV","Commande/Livraison",
    "Resiliation","Support/SAV/Reclamation","Commercial/Offre","Compte/Acces",
    "Annonce/Marketing","Insatisfaction/Colère","Incident/Actualité","Securite/Fraude",
]

def map_mistral_to_coarse(label: str) -> str:
    if not label:
        return "Support/SAV/Reclamation"
    s = label.lower()
    if any(k in s for k in ("factur","paiement","facture")):
        return "Facturation"
    if any(k in s for k in ("reseau","réseau","internet","panne","coupure","hors service")):
        return "Reseau/Internet"
    if any(k in s for k in ("sim","portab","portable","numero","carte sim","portabil")):
        return "Mobile/SIM/Portabilite"
    if any(k in s for k in ("box","tv","degroup","degroupage","décodeur")):
        return "Box/TV"
    if any(k in s for k in ("commande","livraison","colis","suivi","expédition","expedition")):
        return "Commande/Livraison"
    if any(k in s for k in ("résili","resili","resiliation","resiliation")):
        return "Resiliation"
    if any(k in s for k in ("réclam","reclam","sav","sav","réclamation","plainte","reclamation")):
        return "Support/SAV/Reclamation"
    if any(k in s for k in ("commercial","offre","promo","vente")):
        return "Commercial/Offre"
    if any(k in s for k in ("compte","identifiant","mdp","mot de passe","connexion","login","acces")):
        return "Compte/Acces"
    if any(k in s for k in ("annonce","marketing","pub","publicit")):
        return "Annonce/Marketing"
    if any(k in s for k in ("colère","colere","insatisf","insatisfaction","degout","deçu","decu")):
        return "Insatisfaction/Colère"
    if any(k in s for k in ("incident","actualité","actualite","alerte")):
        return "Incident/Actualité"
    if any(k in s for k in ("fraude","secur","sûreté","securite","escroquer","fraud")):
        return "Securite/Fraude"
    # fallback
    return "Support/SAV/Reclamation"
