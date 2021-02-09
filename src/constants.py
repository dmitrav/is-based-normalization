

accuracy_features_names = ['absolute_mass_accuracy_Caffeine_i1_mean', 'absolute_mass_accuracy_Caffeine_i2_mean', 'absolute_mass_accuracy_Caffeine_i3_mean', 'absolute_mass_accuracy_Caffeine_f1_mean',
                           'absolute_mass_accuracy_Fluconazole_i1_mean', 'absolute_mass_accuracy_Fluconazole_i2_mean', 'absolute_mass_accuracy_Fluconazole_i3_mean', 'absolute_mass_accuracy_Fluconazole_f1_mean',
                           'absolute_mass_accuracy_3Heptadecafluorooctylaniline_i1_mean', 'absolute_mass_accuracy_3Heptadecafluorooctylaniline_i2_mean', 'absolute_mass_accuracy_3Heptadecafluorooctylaniline_i3_mean',
                           'absolute_mass_accuracy_Albendazole_i1_mean', 'absolute_mass_accuracy_Albendazole_i2_mean', 'absolute_mass_accuracy_Albendazole_i3_mean', 'absolute_mass_accuracy_Albendazole_f1_mean', 'absolute_mass_accuracy_Albendazole_f2_mean',
                           'absolute_mass_accuracy_Triamcinolone_acetonide_i1_mean', 'absolute_mass_accuracy_Triamcinolone_acetonide_i2_mean', 'absolute_mass_accuracy_Triamcinolone_acetonide_i3_mean', 'absolute_mass_accuracy_Triamcinolone_acetonide_f1_mean', 'absolute_mass_accuracy_Triamcinolone_acetonide_f2_mean',
                           'absolute_mass_accuracy_Perfluorodecanoic_acid_i1_mean', 'absolute_mass_accuracy_Perfluorodecanoic_acid_i2_mean', 'absolute_mass_accuracy_Perfluorodecanoic_acid_i3_mean', 'absolute_mass_accuracy_Perfluorodecanoic_acid_f1_mean',
                           'absolute_mass_accuracy_Tricosafluorododecanoic_acid_i1_mean', 'absolute_mass_accuracy_Tricosafluorododecanoic_acid_i2_mean', 'absolute_mass_accuracy_Tricosafluorododecanoic_acid_i3_mean', 'absolute_mass_accuracy_Tricosafluorododecanoic_acid_f1_mean',
                           'absolute_mass_accuracy_Perfluorotetradecanoic_acid_i1_mean', 'absolute_mass_accuracy_Perfluorotetradecanoic_acid_i2_mean', 'absolute_mass_accuracy_Perfluorotetradecanoic_acid_i3_mean', 'absolute_mass_accuracy_Perfluorotetradecanoic_acid_f1_mean', 'absolute_mass_accuracy_Perfluorotetradecanoic_acid_f2_mean',
                           'absolute_mass_accuracy_Pentadecafluoroheptyl_i1_mean', 'absolute_mass_accuracy_Pentadecafluoroheptyl_i2_mean', 'absolute_mass_accuracy_Pentadecafluoroheptyl_i3_mean']

signal_features_names = [feature_name.replace('absolute_mass_accuracy', 'intensity') for feature_name in accuracy_features_names]