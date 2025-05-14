import xarray as xr
import numpy as np
import os
import yaml
from xseas.stats.normalize import normalize_CMIP6
from xseas import tile_labels, train_perceptron


class SeasonDetect():

    def __init__(self, base_path, config_file):

        """ 
        1. Check if the path exixts and has the right structure (_initialize_path) [to complete]
        2. Check if the config file exists and has the right structure (_load_config) [to complete]
        3. Build the paths for the data, ERA5, CMIP6, clusterings and perceptron model
        """

        self.base_path = self._initialize_path(base_path)
        self.config_file = self._load_config(config_file)
        _ = self._build_paths()

        self.is_clustering_performed = os.path.exists(self.clustering_path)
        self.is_perceptron_trained = os.path.exists(os.path.join(self.perceptron_path,'weights'))
        self.is_projections_classified = os.path.exists(os.path.join(self.base_path, 'projections', self.name + '.nc'))
    

    def _initialize_path(self, path : str) -> str:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Path {path} does not exist.")
        
        # TODO: check if the path has the right structure and automathize folder creation
        return path
    

    def _load_config(self, config_file : str) -> str:
        config_path = os.path.join(self.base_path, 'config', config_file)

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file {config_path} does not exist.")
        
        with open(config_path, 'r') as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
        
        # TODO: check if the config file has the right structure
        try:
            self.name = config['name']
            self.n_seasons = config['n_seasons']
            self.variables = config['variables']
            self.CMIP6_models = config['CMIP6_models']
            self.CMIP6_scenarios = config['CMIP6_scenarios']
            self.n_years_training = config['perceptron_params']['n_years_training']
            self.epochs = config['perceptron_params']['epochs']
        except KeyError as e:
            raise KeyError(f"Missing key in config file: {e}")
        
        return config_path


    def _build_paths(self):
        # Statics paths
        self.data_path = os.path.join(self.base_path, 'data')
        self.ERA5_path = os.path.join(self.data_path, 'ERA5')
        self.CMIP6_path = os.path.join(self.data_path, 'CMIP6')
        self.clustering_path = os.path.join(self.base_path, 'clusterings', f'{self.name}_clust.nc')
        self.perceptron_path = os.path.join(self.base_path, 'perceptron_model', self.name)

        # Dynamic paths for each CMIP model
        for model in self.CMIP6_models:                
            setattr(self, f'{model}_path', os.path.join(self.CMIP6_path, model))

        # Static ERA5 prenormalized path
        self.prenorm_ERA5 = os.path.join(self.ERA5_path, 'prenormalized', 'ERA5_prenorm.nc') if os.path.exists(os.path.join(self.ERA5_path, 'prenormalized', 'ERA5_prenorm.nc')) else False

        # Dynamic CMIP6 prenormalized paths
        for model in self.CMIP6_models:
            setattr(self, f'prenorm_{model}', self.__get_prenorm_path(model)) 

            
    def __get_prenorm_path(self, model):
        """Returns the prenormalized path for a given model if it exists, otherwise returns False."""
        prenorm_path = os.path.join(self.data_path, 'CMIP6', model, 'prenormalized', f'{model}_prenorm.nc')
        return prenorm_path if os.path.exists(prenorm_path) else False
    

    def __repr__(self):
        line = "=" * 60
        clustering_status = "✅ Yes" if self.is_clustering_performed else "❌ No"
        perceptron_status = "✅ Yes" if self.is_perceptron_trained else "❌ No"
        projections_status = "✅ Yes" if self.is_projections_classified else "❌ No"
        prenorm_ERA5_status = "✅ Yes" if os.path.exists(self.prenorm_ERA5) else "❌ No"

        done, undone = [], []
        for model in self.CMIP6_models:
            if getattr(self, f'prenorm_{model}', False):
                done.append(model)
            else:
                undone.append(model)
    
        prenorm_CMIP6_status = "✅ Yes" if len(undone)==0 else "❌ No"


        return (
            f"Project name  : {self.name}\n{line}\n"
            f"📅 Number of Seasons      : {self.n_seasons}\n"
            f"📂 Base Path             : {self.base_path}\n"
            f"📊 Variables ({len(self.variables)})  : {', '.join(self.variables)}\n"
            f"🌀 CMIP6 Models ({len(self.CMIP6_models)}) : {', '.join(self.CMIP6_models)}\n"
            f"📈 CMIP6 Scenarios ({len(self.CMIP6_scenarios)}) : {', '.join(self.CMIP6_scenarios)}\n"
            f"{line}\n"
            f"🔍 Clustering Performed  : {clustering_status}\n"
            f"🤖 Perceptron Trained    : {perceptron_status}\n"
            f"🌐 Projections Classified: {projections_status}\n"
            f"{line}\n"
            f"📊 ERA5 Prenormalized    : {prenorm_ERA5_status}\n"
            f"📊 CMIP6 Prenormalized   : {prenorm_CMIP6_status}\n"
            f"    ✅ Models with prenormalized data    : {', '.join(done) if done else 'None'}\n"
            f"    ❌ Models without prenormalized data : {', '.join(undone) if undone else 'None'}\n"
    )
    






    def prenorm_CMIP6(self):

        for model in self.CMIP6_models:
            if not self.__getattribute__(f'prenorm_{model}'):

                print(f"Prenormalizing {model}...")

                if not os.path.exists(os.path.join(self.base_path, 'data', 'CMIP6', model, 'prenormalized')):
                    os.makedirs(os.path.join(self.base_path, 'data', 'CMIP6', model, 'prenormalized'))
                
                model_path = self.__getattribute__(f'{model}_path')
                dataset_prenorm = normalize_CMIP6(model_path, self.variables, ['tas', 'pr', 'ua', 'va'], self.CMIP6_scenarios)
                dataset_prenorm.to_netcdf(os.path.join(self.base_path, 'data', 'CMIP6', model, 'prenormalized', model + '_prenorm.nc'))
                setattr(self, f'prenorm_{model}', True) 

            else:
                print(f"{model} prenormalized data already exists. Skipping...")

    
    def train_all_models(self, n_years_training=5, epochs=5):

        dataset_train = xr.open_dataset(self.prenorm_ERA5)
        labels = xr.open_dataset(self.clustering_path)
        dataset_train = tile_labels(dataset_train, labels, self.n_seasons)
        array_train  = dataset_train.to_array().values.transpose((2, 3, 1, 0))

        if not os.path.exists(self.perceptron_path):
            os.makedirs(self.perceptron_path)

        mse, r2, models, histories, accuracy = train_perceptron(array_train, len(self.variables), self.perceptron_path, n_years_training, epochs)


        training_metrics = xr.Dataset(
            {
                'mse': (('lat', 'lon'), mse),
                'r2': (('lat', 'lon'), r2),
                'accuracy': (('lat', 'lon'), accuracy)
            },
            coords={
                'lat': dataset_train['lat'],
                'lon': dataset_train['lon']
            }
        )

        if not os.path.exists(os.path.join(self.perceptron_path, 'metrics')):
            os.makedirs(os.path.join(self.perceptron_path, 'metrics'))
        
        training_metrics.to_netcdf(os.path.join(self.perceptron_path, 'metrics', 'training_metrics.nc'))
            
    

