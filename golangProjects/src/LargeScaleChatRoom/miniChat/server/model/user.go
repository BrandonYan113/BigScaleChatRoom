package model

type User struct {
	ID         int    `json:"id"`
	Password   string `json:"password"`
	UserName   string `json:"userName"`
	UserStatus int    `json:"userStatus"`
}
