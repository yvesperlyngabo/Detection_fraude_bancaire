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

if __name__ == "__main__":
    unittest.main()
