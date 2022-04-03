import React, { useState } from 'react'
import { Form, Input, Icon } from 'antd';
import FileUpload from '../../utils/FileUpload';
import Axios from 'axios';

let number = 1;
const { TextArea } = Input;

const Categories = [
    { key: 1, value: "한식" },
    { key: 2, value: "분식" },
    { key: 3, value: "일식" },
    { key: 4, value: "양식" },
    { key: 5, value: "분식" },
    { key: 6, value: "페스트푸드" }
]

const spots = [
    { key: 1, value: "학교 앞" },
    { key: 2, value: "가로수길" }
]

function UploadProductPage(props) {

    const [Title, setTitle] = useState("")
    const [Description, setDescription] = useState("")
    const [Category, setCategory] = useState(1)
    const [Spots, setSpots] = useState(1)
    const [Images, setImages] = useState([])
    const [CallNumber, setCallNumber] = useState("");
    const [reviewlink, setreviewlink] = useState("");


    const titleChangeHandler = (event) => {
        setTitle(event.currentTarget.value)
    }
    
    const handleChangeCallNumber = (event) => {
        console.log(event.currentTarget.value)

        setCallNumber(event.currentTarget.value)
    }

    const handleChangereviewlink = (event) => {
        console.log(event.currentTarget.value)

        setreviewlink(event.currentTarget.value)
    }

    const descriptionChangeHandler = (event) => {
        setDescription(event.currentTarget.value)
    }


    const categoryChangeHandler = (event) => {
        setCategory(event.currentTarget.value)
    }

    const spotChangeHandler = (event) => {
        setSpots(event.currentTarget.value)
    }

    const spotsChangeHandler = (event) => {
        setSpots(event.currentTarget.value)
    }

    const updateImages = (newImages) => {
        setImages(newImages)
    }

    const submitHandler = (event) => {
        event.preventDefault();

        if (!Title || !Description || !Category || !Spots || Images.length === 0) {
            return alert(" 모든 값을 넣어주셔야 합니다.")
        }


        //서버에 채운 값들을 request로 보낸다.

        const body = {
            //로그인 된 사람의 ID 
            writer: props.user.userData._id,
            title: Title,
            description: Description,
            callNumber: CallNumber,
            reviewlink: reviewlink,
            images: Images,
            Categories: Category,
            spots: Spots,
            pCounter: number
        }

        Axios.post('/api/product', body)
            .then(response => {
                if (response.data.success) {
                    alert('상품 업로드에 성공 했습니다.')
                    props.history.push('/')
                } else {
                    alert('상품 업로드에 실패 했습니다.')
                }
            })
            number = number + 1;
    }


    return (
        <div style={{ maxWidth: '700px', margin: '2rem auto' }}>
            <div style={{ textAlign: 'center', marginBottom: '2rem' }}>
                <h2> 음식점 업로드</h2>
            </div>

            <Form onSubmit={submitHandler}>
                {/* DropZone */}
                <FileUpload refreshFunction={updateImages} />

                <br />
                <br />
                <label>이름</label>
                <Input onChange={titleChangeHandler} value={Title} />
                <Icon type="exclamation-circle" style={{ color: 'rgb(256,50,50)' }} />
                
                <label style={{ color: 'rgb(256,50,50)' }}>중복되는 음식점이 있는지 확인 먼저 부탁드립니다.</label>
                
                <br />
                <br />
                <label>음식점 설명</label>
                <TextArea onChange={descriptionChangeHandler} value={Description} />
                <br />
                <br />
                <label>전화번호</label>
                <Input
                    onChange={handleChangeCallNumber}
                    value={CallNumber}
                />
                <br />
                <br />
                <label>리뷰 링크</label>
                <Input
                    onChange={handleChangereviewlink}
                    value={reviewlink}
                />
                <br />
                <br />
                <select onChange={categoryChangeHandler} value={Category}>
                    {Categories.map(item => (
                        <option key={item.key} value={item.key}> {item.value}</option>
                    ))}
                </select>
                <br />
                <br />
                <select onChange={spotChangeHandler} value={Spots}>
                    {spots.map(item => (
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

export default UploadProductPage
