import React, { useState } from 'react';
import { Grid, Paper, Typography, TextField, Container, Button, Divider } from '@mui/material';
import axios from 'axios';


const TextToText = () => {
    const [userInput, setUserInput] = useState('');
    const [response, setResponse] = useState('');
  
    const handleInputChange = (e) => {
      setUserInput(e.target.value);
    };
  
    const handleFormSubmit = async (e) => {
      e.preventDefault();
      try {
        const response = await axios.post('http://127.0.0.1:5000/chat', {
          user_input: userInput,
        });
        setResponse(response.data.response);
        console.log(response);
      } catch (error) {
        console.log(error);
      }
    };

  return (
    <Grid container direction="column">
      <Grid item style={{ marginTop: "2%" }}>
        <Paper elevation={0}>
          <Typography variant='h4'>
            Please input your Question
          </Typography>
          <TextField label="Enter Text" variant="outlined" onChange={handleInputChange} style={{ marginTop: "2%", width: "50%" }} />
          <Container>
            <Button variant='contained' onClick={handleFormSubmit} style={{ marginTop: "2%" }}>
              Submit
            </Button>
            <Divider style={{ margin: "2%" }} />
          </Container>
        </Paper>
      </Grid>
      <Grid item style={{ marginTop: "5%" }}>
        <Paper elevation={0}>
          <Typography variant='h4'>
            Here's your answer
          </Typography>
          <Typography style={{ marginTop: "2%" }}>
            {response && <p>{response}</p>}
          </Typography>
        </Paper>
      </Grid>
    </Grid>
  );
};

export default TextToText;
