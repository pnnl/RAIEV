@echo off

PUSHD %~dp0
SET "SOURCE=%~dp0%"
SET "SOURCE=%SOURCE:\=\\%"

REM Command file for common actions

SET REBUILD=false
IF /I "%2"=="--rebuild" (SET REBUILD=true) ELSE (
	IF NOT "%~2"=="" GOTO rebuild-error
)

IF /I "%1"=="image" GOTO image
IF /I "%1"=="develop" GOTO develop
FOR %%G IN ("lint"
            "test"
            "docs"
            "wheel"
			"requirements"
            "ci") DO (
            IF /I "%1"=="%%~G" GOTO match
)
GOTO arg-error

:image
	docker build -t raiev .
	GOTO end

:develop
	IF %REBUILD%==true CALL make.bat image
	docker run --rm -i -t --mount type=bind,"source=%SOURCE%",target=/raiev raiev bash
	GOTO end

:match
	IF %REBUILD%==true CALL make.bat image
	FOR /F "tokens=* USEBACKQ" %%F IN (`docker run --rm -d -t --mount "type=bind,source=%SOURCE%,target=/raiev" raiev`) DO (
		SET DOCKERID=%%F
	)
	IF DEFINED DOCKERID (
		docker exec %DOCKERID% make "%1"
		docker stop %DOCKERID% >NUL
		SET "DOCKERID="
	)
	GOTO end

:arg-error
	ECHO Unknown first argument "%1". Choose from: develop, lint, fasttest, test, docs, wheel, ci

:rebuild-error
	ECHO Unknown second argument "%2". Set to --rebuild to rebuild the container, or leave blank.

:end
	SET "REBUILD="
	SET "SOURCE="
	POPD
