# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 11:51:17 2017

@author: apezeshk
"""
# Matches case id's (the original weird ones) to xml files from LIDC, s.t. the cases that were missing their
# xml's we can add the appropriate xml file to their folder
import xml.etree.ElementTree as ET
import os
#p1010 ,p0777, p0801, p1009, p1011; The other two didn't match any of the LUNA case id's, so
# can't match them to a seriesInstanceUID
seriesInstUID = ['1.3.6.1.4.1.14519.5.2.1.6279.6001.550599855064600241623943717588',
                 '1.3.6.1.4.1.14519.5.2.1.6279.6001.192256506776434538421891524301',
                 '1.3.6.1.4.1.14519.5.2.1.6279.6001.768276876111112560631432843476',
                 '1.3.6.1.4.1.14519.5.2.1.6279.6001.855232435861303786204450738044',
                 '1.3.6.1.4.1.14519.5.2.1.6279.6001.272123398257168239653655006815']

#These are the two seriesInstanceUID s from LUNA that didn't match any of the LIDC cases we had;
# so use the code below to match them to their xmls, so that we can find out their patient IDs
# >>>The xml's actually don't have the seriesUId s. But using advanced search in LIDC website
# was able to match the missing cases to p0332 (there is another p0332 that we already had) and p1012
#seriesInstUID = ['1.3.6.1.4.1.14519.5.2.1.6279.6001.159996104466052855396410079250',
#                 '1.3.6.1.4.1.14519.5.2.1.6279.6001.153646219551578201092527860224']
                 
pathXML = '/home/apezeshk/Downloads/All-LIDC-xml'
tmp = ''
lstDirsXML = []
for dirName, subdirList, fileList in os.walk(pathXML):
    for filename in fileList:
        if ".xml" in filename.lower():  # check whether the file's DICOM
            if tmp!=dirName:
                tmp=dirName
                lstDirsXML.append(dirName)
                
for dirName in lstDirsXML:
    currentXMLs = os.listdir(dirName)
    for xmlFile in currentXMLs:
        tree = ET.parse(os.path.join(dirName, xmlFile))
        root = tree.getroot()
        currentSeriesInstanceUID = ''
        for child in root:
            #print child.tag
            if 'ResponseHeader' in child.tag:
                for gchild in child:
                    #the gchild of interest is sth like {http://www.nih.gov}SeriesInstanceUid
                    #so add the '}' s.t. other strings with SeriesInstanceUid in them wouldn't
                    #trigger the if.
                    if ('}SeriesInstanceUid' in gchild.tag) or ('}SeriesInstanceUID' in gchild.tag):                        
                        currentSeriesInstanceUID = gchild.text
                        break                

            
        if currentSeriesInstanceUID in seriesInstUID:
            print(currentSeriesInstanceUID + ' : ' + os.path.join(dirName, xmlFile))
            
        

        
        
        

         
                
