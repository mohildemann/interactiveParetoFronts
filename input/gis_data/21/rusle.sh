#!/bin/bash
g.proj -c epsg=32637
r.in.gdal input=C:/Users/morit/AppData/Local/Temp/grassdata/mytemploc_utm32n/dem_reprojected.tif band=1 output=demraster --overwrite -o
r.in.gdal input=C:/Users/morit/AppData/Local/Temp/grassdata/mytemploc_utm32n/landuse_raster.tif band=1 output=landuse_ras --overwrite -o
r.in.gdal input=C:/Users/morit/AppData/Local/Temp/grassdata/mytemploc_utm32n/r_factor_realizations/r_factor_0_clipped.tif band=1 output=rfactor_ras --overwrite -o
r.in.gdal input=C:/Users/morit/AppData/Local/Temp/grassdata/mytemploc_utm32n/k_factor_realizations/k_factor_r_0clipped.tif band=1 output=kfactor_ras --overwrite -o
r.in.gdal input=C:/Users/morit/AppData/Local/Temp/grassdata/mytemploc_utm32n/c_factor_clipped.tif band=1 output=cfactor_ras --overwrite -o
r.in.gdal input=C:/Users/morit/AppData/Local/Temp/grassdata/mytemploc_utm32n/Slope_degrees.tif band=1 output=slope_degrees_ras --overwrite -o
v.in.ogr min_area=0.0 snap=-1.0 input=C:/Users/morit/AppData/Local/Temp/grassdata/mytemploc_utm32n/contour_lines_with_watershed_id_enerata.shp layer="contour_lines_with_watershed_id_enerata" output="contour_lines" --overwrite -o
v.in.ogr min_area=0.0 snap=-1.0 input=C:/Users/morit/AppData/Local/Temp/grassdata/mytemploc_utm32n/Basins_4.shp layer="Basins_4" output=watersheds_shp --overwrite -o
g.region n=1156020.7409 s=1145261.853 e=364958.1745 w=357194.6588 res=30.565022451815462
v.to.rast input=contour_lines layer="contour_lines_with_watershed_id_enerata" type="point,line,area" where="pos_rank in (9, 10, 18, 19, 24, 26, 30, 32, 33, 36, 37, 42, 44, 45, 50, 51, 53, 54, 59, 62, 63, 64, 65, 66, 69, 72, 76, 78, 80, 84, 86, 92, 96, 100, 102, 103, 108, 111, 116, 119, 121, 127, 129, 135, 146, 148, 150, 154, 156, 164, 165, 168, 170, 173, 177, 179, 180, 181, 182, 192, 194, 196, 197, 200, 203, 204, 205, 207, 216, 219, 220, 225, 226, 229, 231, 232, 238, 239, 240, 245, 247, 249, 252, 254, 257, 258, 266, 282)" use="val" value=1 memory=300 output=selected_contour_lines_ras --overwrite
v.to.rast input=watersheds_shp layer="Basins_4" type="point,line,area" where="pos_rank in (9, 10, 18, 19, 24, 26, 30, 32, 33, 36, 37, 42, 44, 45, 50, 51, 53, 54, 59, 62, 63, 64, 65, 66, 69, 72, 76, 78, 80, 84, 86, 92, 96, 100, 102, 103, 108, 111, 116, 119, 121, 127, 129, 135, 146, 148, 150, 154, 156, 164, 165, 168, 170, 173, 177, 179, 180, 181, 182, 192, 194, 196, 197, 200, 203, 204, 205, 207, 216, 219, 220, 225, 226, 229, 231, 232, 238, 239, 240, 245, 247, 249, 252, 254, 257, 258, 266, 282)" use="val" value=1 memory=300 output=protected_watersheds_ras --overwrite
r.null map=selected_contour_lines_ras null=0
r.flow  elevation=demraster barrier=selected_contour_lines_ras flowline=flowlineraster flowlength=flowlengthraster flowaccumulation=flowaccraster --overwrite
g.region raster=flowlengthraster
r.mapcalc --overwrite expression=""pfactor_ras" = if(landuse_ras == 1, (if(!isnull(protected_watersheds_ras), 0.7 , 0.2)),(if(landuse_ras == 2, if(!isnull(protected_watersheds_ras), 0.7 , 0.2), (if(landuse_ras == 3 || landuse_ras == 4 || landuse_ras == 5, 1,0)) )) )"
r.mapcalc --overwrite expression=""m_ras" = (((sin(slope_degrees_ras * 3.141592653589793/180) / 0.0896)/ (3 + sin(slope_degrees_ras * 3.141592653589793/180) * 0.8 + 0.56))/(1+ (sin(slope_degrees_ras * 3.141592653589793/180) / 0.0896)/(3 + sin(slope_degrees_ras * 3.141592653589793/180) * 0.8 + 0.56)))"
r.mapcalc --overwrite expression=""lfactor_ras"=(flowlengthraster/22.13)^m_ras"
r.mapcalc --overwrite expression=""sfactor_ras"=(if(slope_degrees_ras<5,10 * sin(slope_degrees_ras) + 0.03,(if(5<slope_degrees_ras<=10,16*sin(slope_degrees_ras)-0.55,21.9*sin(slope_degrees_ras)-0.96))))"
r.mapcalc --overwrite expression=""lsfactor_ras"=lfactor_ras*sfactor_ras"
r.mapcalc --overwrite expression=""rusle_ras"=rfactor_ras*kfactor_ras*lsfactor_ras*cfactor_ras*pfactor_ras"
r.univar -t map=rusle_ras separator=comma output="C:\Users\morit\AppData\Local\Temp\grassdata\3255ab2a58817c5184a5a00ad9582a7f\21\rusle_total.csv" --overwrite
r.out.gdal -t -m input=flowlengthraster output="C:\Users\morit\AppData\Local\Temp\grassdata\3255ab2a58817c5184a5a00ad9582a7f\21/flowlength.tif" format=GTiff  createopt=TFW=YES,COMPRESS=LZW --overwrite
r.out.gdal -t -m input=rfactor_ras output="C:\Users\morit\AppData\Local\Temp\grassdata\3255ab2a58817c5184a5a00ad9582a7f\21/r_factor.tif" format=GTiff  createopt=TFW=YES,COMPRESS=LZW --overwrite
r.out.gdal -t -m input=kfactor_ras output="C:\Users\morit\AppData\Local\Temp\grassdata\3255ab2a58817c5184a5a00ad9582a7f\21/k_factor.tif" format=GTiff  createopt=TFW=YES,COMPRESS=LZW --overwrite
r.out.gdal -t -m input=lfactor_ras output="C:\Users\morit\AppData\Local\Temp\grassdata\3255ab2a58817c5184a5a00ad9582a7f\21/l_factor.tif" format=GTiff  createopt=TFW=YES,COMPRESS=LZW --overwrite
r.out.gdal -t -m input=sfactor_ras output="C:\Users\morit\AppData\Local\Temp\grassdata\3255ab2a58817c5184a5a00ad9582a7f\21/s_factor.tif" format=GTiff  createopt=TFW=YES,COMPRESS=LZW --overwrite
r.out.gdal -t -m input=lsfactor_ras output="C:\Users\morit\AppData\Local\Temp\grassdata\3255ab2a58817c5184a5a00ad9582a7f\21/ls_factor.tif" format=GTiff  createopt=TFW=YES,COMPRESS=LZW --overwrite
r.out.gdal -t -m input=pfactor_ras output="C:\Users\morit\AppData\Local\Temp\grassdata\3255ab2a58817c5184a5a00ad9582a7f\21/p_factor.tif" format=GTiff  createopt=TFW=YES,COMPRESS=LZW --overwrite
r.out.gdal -t -m input=cfactor_ras output="C:\Users\morit\AppData\Local\Temp\grassdata\3255ab2a58817c5184a5a00ad9582a7f\21/c_factor.tif" format=GTiff  createopt=TFW=YES,COMPRESS=LZW --overwrite
r.out.gdal -t -m input=rusle_ras output="C:\Users\morit\AppData\Local\Temp\grassdata\3255ab2a58817c5184a5a00ad9582a7f\21/rusle.tif" format=GTiff  createopt=TFW=YES,COMPRESS=LZW --overwrite
v.extract input=contour_lines  output=selected_contour_lines_shp layer="contour_lines_with_watershed_id_enerata" where="pos_rank in (9, 10, 18, 19, 24, 26, 30, 32, 33, 36, 37, 42, 44, 45, 50, 51, 53, 54, 59, 62, 63, 64, 65, 66, 69, 72, 76, 78, 80, 84, 86, 92, 96, 100, 102, 103, 108, 111, 116, 119, 121, 127, 129, 135, 146, 148, 150, 154, 156, 164, 165, 168, 170, 173, 177, 179, 180, 181, 182, 192, 194, 196, 197, 200, 203, 204, 205, 207, 216, 219, 220, 225, 226, 229, 231, 232, 238, 239, 240, 245, 247, 249, 252, 254, 257, 258, 266, 282)"
v.extract input=watersheds_shp  output=selected_watersheds_shp layer="Basins_4" where="pos_rank in (9, 10, 18, 19, 24, 26, 30, 32, 33, 36, 37, 42, 44, 45, 50, 51, 53, 54, 59, 62, 63, 64, 65, 66, 69, 72, 76, 78, 80, 84, 86, 92, 96, 100, 102, 103, 108, 111, 116, 119, 121, 127, 129, 135, 146, 148, 150, 154, 156, 164, 165, 168, 170, 173, 177, 179, 180, 181, 182, 192, 194, 196, 197, 200, 203, 204, 205, 207, 216, 219, 220, 225, 226, 229, 231, 232, 238, 239, 240, 245, 247, 249, 252, 254, 257, 258, 266, 282)"
v.out.ogr input=selected_contour_lines_shp output="C:\Users\morit\AppData\Local\Temp\grassdata\3255ab2a58817c5184a5a00ad9582a7f\21/terraces.geojson" format=GeoJSON  --overwrite
v.out.ogr input=selected_watersheds_shp output="C:\Users\morit\AppData\Local\Temp\grassdata\3255ab2a58817c5184a5a00ad9582a7f\21/protected_watersheds.geojson" format=GeoJSON --overwrite
