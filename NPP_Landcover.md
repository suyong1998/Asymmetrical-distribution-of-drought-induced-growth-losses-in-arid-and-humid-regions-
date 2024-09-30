```javascript
var points = ee.FeatureCollection('projects/ee-yongsu/assets/ITRDB_site_FZS');
var npp_dataset = ee.ImageCollection('MODIS/061/MOD17A3HGF')
    .filterDate('2001-01-01', '2022-12-31');

var npp_image = npp_dataset.toBands();
var npp_values = npp_image.sampleRegions({
  collection: points,
  scale: 500,  
  geometries:true
});
print('npp_values',npp_values.limit(5))
Export.table.toDrive({
    collection: npp_values,
    description: 'npp_values',
    fileFormat: 'CSV'
});

var imageLayers =  ee.Image("NASA/ASTER_GED/AG100_003").select('elevation')
                    .addBands(ee.Image(ee.ImageCollection("ESA/WorldCover/v100").first()))
                    .rename(['DEM','LandCver'])
                        
var points = ee.FeatureCollection('projects/ee-yongsu/assets/ITRDB_site_FZS');

var dataset = ee.ImageCollection('ESA/WorldCover/v100').first();

var visualization = {
  bands: ['Map'],
};

Map.centerObject(dataset);

Map.addLayer(dataset, visualization, 'Landcover');
Map.addLayer(points)
var demValues = imageLayers.sampleRegions({
  collection: points,
  scale: 10,  
  geometries:true
});
print('sample_points',demValues.limit(10))
// 
Export.table.toDrive({
  collection: demValues,
  description: 'itrdb_dem_landconr_values',
  fileFormat: 'CSV'
});


```

