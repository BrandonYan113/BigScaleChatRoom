package main

import (
	"fmt"
	"redisLearn/miniChat/client/process"
)

var (
	id           int
	psw          string
	pswConfirmed string
	userName     string
)

func main() {
	var key int
	loop := true

	for {
		if !loop {
			fmt.Println("exit chat room")
			break
		}
		fmt.Println("--------------welcome to online chat room--------------")
		fmt.Println("1 Login")
		fmt.Println("2 register")
		fmt.Println("3 exit")
		fmt.Scanf("%d\n", &key)
		switch key {
		case 1:
			fmt.Println("input user ID")
			fmt.Scanf("%d\n", &id)
			fmt.Println("input password")
			fmt.Scanf("%s\n", &psw)
			userPrc := &process.UserProcess{}
			err := userPrc.Login(id, psw)
			if err != nil {
				fmt.Printf("ID(%d) or psw(%s) err: %v \n", id, psw, err)
				key = 1
				continue
			}

		case 2:
			fmt.Println("input User name(nick name)")
			fmt.Scanln(&userName)
			fmt.Println("input user ID")
			fmt.Scanln(&id)
			fmt.Println("input password")
			fmt.Scanln(&psw)
			fmt.Println("verify input password")
			fmt.Scanln(&pswConfirmed)
			if psw != pswConfirmed {
				key = 2
				fmt.Println("password not equal verify password, please input it again")
				continue
			}
			userPrc := &process.UserProcess{}
			err := userPrc.Register(userName, id, psw)
			if err != nil {
				fmt.Println("Register failed")
			}

		case 3:
			fmt.Println("exit")
			loop = false
		default:
			fmt.Println("not an options, please input a number in 1-3")
		}
	}
	if key == 1 {

	} else if key == 2 {

	}
}
