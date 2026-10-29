import unittest
import pandas as pd
import numpy as np
from model.preprocessing import nettoyer_et_enrichir, encoder_features, preparer_transaction

class TestPreprocessing(unittest.TestCase):
    def setUp(self):
        self.df_test = pd.DataFrame({
            "ID Clients": ["C1", "C1", "C2"],
            "Numero de compte": ["A1", "A1", "A2"],
            "Identifiant operation": ["OP1", "OP2", "OP3"],
            "Type de transaction": ["ATM", "Paiement en ligne", "ATM"],
            "Status operation": ["Validé", "Validé", "Echoué"],
            "Localisation": ["Dakar", "Dakar", "Thiès"],
            "Date": ["2023-01-01 10:00:00", "2023-01-02 11:00:00", "2023-01-01 23:00:00"],
            "Montant": [5000, 10000, 20000],
            "Target": ["Normal", "Normal", "Fraude"]
        })

    def test_nettoyer_et_enrichir(self):
        df_clean, top_loc = nettoyer_et_enrichir(self.df_test, nb_localisations=1)

        # Test colonnes temporelles
        self.assertIn("heure", df_clean.columns)
        self.assertIn("est_nuit", df_clean.columns)
        self.assertEqual(df_clean.iloc[2]["est_nuit"], 1) # 23h

        # Test variables comportementales
        self.assertIn("montant_moyen_7j", df_clean.columns)
        self.assertIn("nb_trans_7j", df_clean.columns)

        # Test regroupement localisation
        self.assertEqual(len(top_loc), 1)
        self.assertIn("Autre", df_clean["Localisation_grp"].values)

    def test_encoder_features(self):
        df_clean, _ = nettoyer_et_enrichir(self.df_test)
        X, cols = encoder_features(df_clean)

        self.assertIsInstance(X, pd.DataFrame)
        self.assertTrue(all(c in X.columns for c in ["log_montant", "heure", "montant_moyen_7j"]))

    def test_variables_7j_passe_seulement(self):
        """Les variables 7 jours n'utilisent que les transactions passées du client."""
        df_clean, _ = nettoyer_et_enrichir(self.df_test)
        c1 = df_clean[df_clean["ID Clients"] == "C1"].reset_index(drop=True)

        # Première transaction : aucun historique
        self.assertEqual(c1.loc[0, "nb_trans_7j"], 0)
        self.assertEqual(c1.loc[0, "montant_moyen_7j"], 5000)  # repli sur le montant courant

        # Deuxième transaction : seule la première est dans la fenêtre
        self.assertEqual(c1.loc[1, "nb_trans_7j"], 1)
        self.assertEqual(c1.loc[1, "montant_moyen_7j"], 5000)

    def test_ordre_des_lignes_conserve(self):
        """Le prétraitement ne réordonne pas les lignes (découpages train/test reproductibles)."""
        df = self.df_test.iloc[[2, 0, 1]].reset_index(drop=True)   # ordre volontairement mélangé
        df_clean, _ = nettoyer_et_enrichir(df)
        self.assertEqual(list(df_clean["Identifiant operation"]), ["OP3", "OP1", "OP2"])
        # et les valeurs restent attachées à la bonne ligne
        ligne_op2 = df_clean[df_clean["Identifiant operation"] == "OP2"].iloc[0]
        self.assertEqual(ligne_op2["nb_trans_7j"], 1)

    def test_variables_7j_hors_fenetre(self):
        """Une transaction de plus de 7 jours n'entre plus dans la fenêtre."""
        df = self.df_test.copy()
        df.loc[1, "Date"] = "2023-01-20 11:00:00"
        df_clean, _ = nettoyer_et_enrichir(df)
        c1 = df_clean[df_clean["ID Clients"] == "C1"].reset_index(drop=True)
        self.assertEqual(c1.loc[1, "nb_trans_7j"], 0)

    def test_preparer_transaction(self):
        df_clean, top_loc = nettoyer_et_enrichir(self.df_test)
        _, feature_columns = encoder_features(df_clean)

        transaction = {
            "Type de transaction": "ATM",
            "Status operation": "Validé",
            "Localisation": "Dakar",
            "Date": "2023-01-03 12:00:00",
            "Montant": 7000,
            "nb_transactions_client": 5,
            "montant_moyen_7j": 6000,
            "nb_trans_7j": 2
        }

        df_ligne = preparer_transaction(transaction, top_loc, feature_columns)
        self.assertEqual(df_ligne.shape[0], 1)
        self.assertEqual(df_ligne.shape[1], len(feature_columns))
        self.assertAlmostEqual(df_ligne["log_montant"].iloc[0], np.log1p(7000))

    def test_preparer_transaction_sans_historique(self):
        """Sans historique fourni, les défauts correspondent à ceux de l'entraînement."""
        df_clean, top_loc = nettoyer_et_enrichir(self.df_test)
        _, feature_columns = encoder_features(df_clean)
        transaction = {
            "Type de transaction": "ATM", "Status operation": "Validé",
            "Localisation": "Dakar", "Date": "2023-01-03 12:00:00", "Montant": 7000,
        }
        df_ligne = preparer_transaction(transaction, top_loc, feature_columns)
        self.assertEqual(df_ligne["nb_trans_7j"].iloc[0], 0)
        self.assertEqual(df_ligne["montant_moyen_7j"].iloc[0], 7000)

if __name__ == "__main__":
    unittest.main()
