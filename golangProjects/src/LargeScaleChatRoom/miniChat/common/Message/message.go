package Message

import (
	"redisLearn/miniChat/server/model"
)

const (
	LoginMesType            = "logMes"
	LoginReturnMesType      = "logResMes"
	RegisterMesType         = "registerMes"
	RegisterReturnMesType   = "regsRtMes"
	NotifyUserStatusMesType = "NotifyUserStatusMes"
	SmsMessageType          = "SMSMes"
)

// 用户状态常量设置

const (
	UserOnlineStatus = iota
	UserOfflineStatus
	UserBusyStatus
)

type Message struct {
	Type string
	Data string
}

type LoginMes struct {
	ID       int    `json:"id"`
	Password string `json:"password"`
	UserName string `json:"userName"`
}

type LoginReturnMes struct {
	Code  int    `json:"code"`  //状态码 500表示用户未注册，200表示登录成功
	Users []int  `json:"users"` // 用户在线id列表
	Err   string `json:"error"` //错误信息，nil无错误，否则有错误
}

type RegisterMes struct {
	User model.User `json:"user"`
}

type RegisterResponseMes struct {
	Code  int    `json:"code"` // 400表示已经占有，200注册成功
	Error string `json:"error"`
}

type NotifyUserStatusMes struct {
	UserId int `json:"userId"`
	Status int `json:"status"` //用户状态
}

type SmsMes struct {
	Content string `json:"content"`
	User    *User  `json:"user"`
}
