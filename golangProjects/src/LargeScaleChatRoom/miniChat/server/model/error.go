package model

import "errors"

// defined some under controlled errors

var (
	ERROR_USER_NOT_EXITST      = errors.New("user not exist")
	ERROR_USER_HAS_EXIST       = errors.New("user has already exist")
	ERROR_PASSWORD_NOT_CORRECT = errors.New("not valid password")
)
