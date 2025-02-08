:: filepath: /C:/Users/brahm/dev/ik/capstone/ShopTalk-03-Feb-25/download_image.bat
@echo off
setlocal

:: URL of the image
#t IMAGE_URL=https://unsplash.com/photos/green-fabric-sofa-fZuleEfeA1Q?utm_content=creditShareLink&utm_medium=referral&utm_source=unsplash
#et IMAGE_URL=https://unsplash.com/photos/green-fabric-sofa-fZuleEfeA1Q?utm_content=creditShareLink&utm_medium=referral&utm_source=unsplash
set IMAGE_URL=https://unsplash.com/photos/person-taking-photo-of-grey-padded-chair-inside-room-ItMggD0EguY?utm_content=creditShareLink&utm_medium=referral&utm_source=unsplash

:: Output file name
set OUTPUT_FILE=downloaded_image02.g

:: Use curl to download the image
curl -L -o %OUTPUT_FILE% %IMAGE_URL%

echo Image downloaded as %OUTPUT_FILE%
endlocal