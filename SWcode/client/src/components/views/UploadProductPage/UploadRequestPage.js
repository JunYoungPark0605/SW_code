import React, { useState } from 'react'
import { Typography, Button, Form, Input, Icon } from 'antd';
import FileUpload from '../../utils/FileUpload';
import Axios from 'axios';
import moment from "moment";
import { Formik } from 'formik';
import * as Yup from 'yup';
import { registerUser } from "../../../_actions/user_actions";
import { useDispatch } from "react-redux";

const { TextArea } = Input;

const RequestCategories = [
    { key: 1, value: "삭제" },
    { key: 2, value: "수정" }
]


function UploadRequestPage(props) {

    const [Title, setTitle] = useState("")
    const [Description, setDescription] = useState("")
    const [RequestCategory, setRequestCategory] = useState(1)

    const titleChangeHandler = (event) => {
        setTitle(event.currentTarget.value)
    }

    const descriptionChangeHandler = (event) => {
        setDescription(event.currentTarget.value)
    }


    const requestCategoryChangeHandler = (event) => {
        setRequestCategory(event.currentTarget.value)
    }

    const submitHandler = (event) => {
        event.preventDefault();

        if (!Title || !Description || !RequestCategory) {
            return alert(" 모든 값을 넣어주셔야 합니다.")
        }


        //서버에 채운 값들을 request로 보낸다.

        const body = {
            //로그인 된 사람의 ID 
            writer: props.user.userData._id,
            title: Title,
            description: Description,
            requestCategories: RequestCategory
        }

        Axios.post('/api/request', body)
            .then(response => {
                if (response.data.success) {
                    alert('상품 업로드에 성공 했습니다.')
                    props.history.push('/')
                } else {
                    alert('상품 업로드에 실패 했습니다.')
                }
            })
    }


    return (
        <div style={{ maxWidth: '700px', margin: '2rem auto' }}>
            <div style={{ textAlign: 'center', marginBottom: '2rem' }}>
                <h2> 수정 및 삭제 요청</h2>
            </div>

            <Form onSubmit={submitHandler}>
                {/* DropZone */}

                <br />
                <br />
                <label>제목</label>
                <Input onChange={titleChangeHandler} value={Title} />
                <Icon type="exclamation-circle" style={{ color: 'rgb(256,50,50)' }} />
                
                <label style={{ color: 'rgb(256,50,50)' }}> (요청할 게시글 이름)_(수정/삭제)요청합니다.</label>
                
                <br />
                <br />
                <label>요청 사항</label>
                <TextArea onChange={descriptionChangeHandler} value={Description} />
                <br />
                <br />
                <br />
                <br />
                <select onChange={requestCategoryChangeHandler} value={RequestCategory}>
                    {RequestCategories.map(item => (
                        <option key={item.key} value={item.key}> {item.value}</option>
                    ))}
                </select>
                <br />
                <br />
                <button type="submit">
                    확인
                </button>
            </Form>


        </div>
    )
}

export default UploadRequestPage
