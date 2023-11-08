package model

import (
	"encoding/json"
	"fmt"
	"github.com/gomodule/redigo/redis"
)

var (
	GlobaluserDo *userDo
)

// handle with information from user struct

type userDo struct {
	Pool *redis.Pool
}

// return a userDo struct by using factory mode

func NewUserDo(pool *redis.Pool) (user *userDo) {
	user = &userDo{
		Pool: pool,
	}
	return
}

func (p *userDo) getUserById(conn redis.Conn, id int) (user *User, err error) {
	res, err := redis.String(conn.Do("HGet", "users", id))
	if err != nil {
		if err == redis.ErrNil {
			err = ERROR_USER_NOT_EXITST
		}
		return
	}
	err = json.Unmarshal([]byte(res), &user)
	if err != nil {
		fmt.Println("unmarshal redis byte err: ", err)
		return
	}
	return
}

// check Login information

func (p *userDo) HandleLogin(userId int, psw string) (user *User, err error) {
	conn := p.Pool.Get()
	user, err = p.getUserById(conn, userId)
	if err != nil {
		return nil, err
	}
	if user.Password == psw {
		return
	}
	return nil, ERROR_PASSWORD_NOT_CORRECT
}

func (p *userDo) HandleRegister(user *User) (err error) { //create user
	conn := p.Pool.Get() //连接Redis
	defer conn.Close()
	_, err = p.getUserById(conn, user.ID)
	if err == nil {
		return ERROR_USER_HAS_EXIST
	}
	serialData, err := json.Marshal(user)
	if err != nil {
		return
	}
	_, err = conn.Do("HSet", "users", user.ID, string(serialData))
	if err != nil {
		fmt.Println("redis store register user error: ", err)
	}
	return
}
